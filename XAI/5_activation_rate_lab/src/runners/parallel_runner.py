from functools import partial
from multiprocessing import Pipe, Process
from .helpers import (
	rware_obs_to_atoms,
	_RWARE_ACTION_MAP,
	get_theory_dict,
)

import numpy as np
import pandas as pd

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2
import os

_ROOT = os.getcwd()


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
	def __init__(self, args, logger):
		self.args = args
		self.logger = logger
		self.batch_size = self.args.batch_size_run

		self.current_time = args.current_time
		if args.explanations:
			self._example_path = os.path.join(
				_ROOT,
				"plots",
			)
			os.makedirs(self._example_path, exist_ok=True)

		# Make subprocesses for the envs
		self.parent_conns, self.worker_conns = zip(*[
			Pipe() for _ in range(self.batch_size)
		])

		# registering both smac and smacv2 causes a pysc2 error
		# --> dynamically register the needed env
		if self.args.env == "sc2":
			register_smac()
		elif self.args.env == "sc2v2":
			register_smacv2()

		env_fn = env_REGISTRY[self.args.env]
		env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
		for i in range(self.batch_size):
			env_args[i]["seed"] += i
			env_args[i]["common_reward"] = self.args.common_reward
			env_args[i]["reward_scalarisation"] = self.args.reward_scalarisation
		self.ps = [
			Process(
				target=env_worker,
				args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
			)
			for env_arg, worker_conn in zip(env_args, self.worker_conns)
		]

		for p in self.ps:
			p.daemon = True
			p.start()

		self.parent_conns[0].send(("get_env_info", None))
		self.env_info = self.parent_conns[0].recv()
		self.episode_limit = self.env_info["episode_limit"]

		self.t = 0

		self.t_env = 0

		self.train_returns = []
		self.test_returns = []
		self.train_stats = {}
		self.test_stats = {}

		self.log_train_stats_t = -100000

	def setup(self, scheme, groups, preprocess, mac):
		self.new_batch = partial(
			EpisodeBatch,
			scheme,
			groups,
			self.batch_size,
			self.episode_limit + 1,
			preprocess=preprocess,
			device=self.args.device,
		)
		self.mac = mac
		self.scheme = scheme
		self.groups = groups
		self.preprocess = preprocess

	def get_env_info(self):
		return self.env_info

	def save_replay(self):
		self.parent_conns[0].send(("save_replay", None))

	def close_env(self):
		for parent_conn in self.parent_conns:
			parent_conn.send(("close", None))

	def reset(self):
		self.batch = self.new_batch()

		# Reset the envs
		for parent_conn in self.parent_conns:
			parent_conn.send(("reset", None))

		pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
		# Get the obs, state and avail_actions back
		for parent_conn in self.parent_conns:
			data = parent_conn.recv()
			pre_transition_data["state"].append(data["state"])
			pre_transition_data["avail_actions"].append(data["avail_actions"])
			pre_transition_data["obs"].append(data["obs"])

		self.batch.update(pre_transition_data, ts=0)

		self.t = 0
		self.env_steps_this_run = 0

	def run(self, test_mode=False, explanation_mode=False):
		self.reset()

		all_terminated = False
		if self.args.common_reward:
			episode_returns = [0 for _ in range(self.batch_size)]
		else:
			episode_returns = [
				np.zeros(self.args.n_agents) for _ in range(self.batch_size)
			]
		episode_lengths = [0 for _ in range(self.batch_size)]
		self.mac.init_hidden(batch_size=self.batch_size)
		terminated = [False for _ in range(self.batch_size)]
		envs_not_terminated = [
			b_idx for b_idx, termed in enumerate(terminated) if not termed
		]
		final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

		while True:
			# Pass the entire batch of experiences up till now to the agents
			# Receive the actions for each agent at this timestep in a batch for each un-terminated env
			actions = self.mac.select_actions(
				self.batch,
				t_ep=self.t,
				t_env=self.t_env,
				bs=envs_not_terminated,
				test_mode=test_mode,
			)
			cpu_actions = actions.to("cpu").numpy()

			# Update the actions taken
			actions_chosen = {"actions": actions.unsqueeze(1)}
			self.batch.update(
				actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
			)

			# Send actions to each env
			action_idx = 0
			for idx, parent_conn in enumerate(self.parent_conns):
				if idx in envs_not_terminated:  # We produced actions for this env
					if not terminated[
						idx
					]:  # Only send the actions to the env if it hasn't terminated
						parent_conn.send(("step", cpu_actions[action_idx]))
					action_idx += 1  # actions is not a list over every env
					if idx == 0 and test_mode and self.args.render:
						parent_conn.send(("render", None))

			# Update envs_not_terminated
			envs_not_terminated = [
				b_idx for b_idx, termed in enumerate(terminated) if not termed
			]
			all_terminated = all(terminated)
			if all_terminated:
				break

			# Post step data we will insert for the current timestep
			post_transition_data = {"reward": [], "terminated": []}
			# Data for the next step we will insert in order to select an action
			pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

			# Receive data back for each unterminated env
			for idx, parent_conn in enumerate(self.parent_conns):
				if not terminated[idx]:
					data = parent_conn.recv()
					# Remaining data for this current timestep
					post_transition_data["reward"].append((data["reward"],))

					episode_returns[idx] += data["reward"]
					episode_lengths[idx] += 1
					if not test_mode:
						self.env_steps_this_run += 1

					env_terminated = False
					if data["terminated"]:
						final_env_infos.append(data["info"])
					if data["terminated"] and not data["info"].get(
						"episode_limit", False
					):
						env_terminated = True
					terminated[idx] = data["terminated"]
					post_transition_data["terminated"].append((env_terminated,))

					# Data for the next timestep needed to select an action
					pre_transition_data["state"].append(data["state"])
					pre_transition_data["avail_actions"].append(data["avail_actions"])
					pre_transition_data["obs"].append(data["obs"])

			# Add post_transiton data into the batch
			self.batch.update(
				post_transition_data,
				bs=envs_not_terminated,
				ts=self.t,
				mark_filled=False,
			)

			# Move onto the next timestep
			self.t += 1

			# Add the pre-transition data
			self.batch.update(
				pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
			)

		if not test_mode:
			self.t_env += self.env_steps_this_run

		# Get stats back for each env
		for parent_conn in self.parent_conns:
			parent_conn.send(("get_stats", None))

		env_stats = []
		for parent_conn in self.parent_conns:
			env_stat = parent_conn.recv()
			env_stats.append(env_stat)

		cur_stats = self.test_stats if test_mode else self.train_stats
		cur_returns = self.test_returns if test_mode else self.train_returns
		log_prefix = "test_" if test_mode else ""
		infos = [cur_stats] + final_env_infos
		cur_stats.update({
			k: sum(d.get(k, 0) for d in infos)
			for k in set.union(*[set(d) for d in infos])
		})
		cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
		cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

		cur_returns.extend(episode_returns)

		n_test_runs = (
			max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
		)
		if explanation_mode:
			try:
				current_env_tested = self.args.env_args["key"].split(":")[1]
			except:
				current_env_tested = self.args.env_args["key"]

			env_obs_to_atom_fn = partial(
				rware_obs_to_atoms,
				sensor_range=self.env_info["sensor_range"],
			)
			env_action_map = _RWARE_ACTION_MAP

			theory_dicts = self.load_theory_dicts(env_action_map)

			# Save header to file
			path_to_save = os.path.join(
				self._example_path, f"activation_rate_{self.current_time}.csv"
			)
			if not os.path.exists(path_to_save):
				with open(path_to_save, "w") as file_write:
					file_write.write(
						"index,seed,env_theory,env_test,steps,agent,action,activation_rate\n"
					)
			for agent_idx in range(self.env_info["n_agents"]):
				### -------- Activation rate
				activation_rate(
					self.batch.data.transition_data,
					self.env_info["original_obs_shape"],
					self.args.theory_index,
					current_env_tested,
					current_env_tested,
					self.args.load_step,
					env_obs_to_atom_fn,
					theory_dicts,
					env_action_map,
					agent_idx,
					path_to_save,
				)

		if test_mode and (len(self.test_returns) == n_test_runs):
			self._log(cur_returns, cur_stats, log_prefix)
		elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
			self._log(cur_returns, cur_stats, log_prefix)
			if hasattr(self.mac.action_selector, "epsilon"):
				self.logger.log_stat(
					"epsilon", self.mac.action_selector.epsilon, self.t_env
				)
			self.log_train_stats_t = self.t_env

		return self.batch

	def load_theory_dicts(self, action_map, theory_step=None):
		if theory_step is None:
			theory_step = self.args.load_step
		theory_dicts = get_theory_dict(
			theory_step,
			self.args.theory_index,
			n_agents=self.env_info["n_agents"],
			action_map=action_map,
		)
		assert np.sum([[len(v) for v in th.values()] for th in theory_dicts]) > 0, (
			"Theories are all empty"
		)

		return theory_dicts

	def _log(self, returns, stats, prefix):
		if self.args.common_reward:
			self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
			self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
		else:
			for i in range(self.args.n_agents):
				self.logger.log_stat(
					prefix + f"agent_{i}_return_mean",
					np.array(returns)[:, i].mean(),
					self.t_env,
				)
				self.logger.log_stat(
					prefix + f"agent_{i}_return_std",
					np.array(returns)[:, i].std(),
					self.t_env,
				)
			total_returns = np.array(returns).sum(axis=-1)
			self.logger.log_stat(
				prefix + "total_return_mean", total_returns.mean(), self.t_env
			)
			self.logger.log_stat(
				prefix + "total_return_std", total_returns.std(), self.t_env
			)
		returns.clear()

		for k, v in stats.items():
			if k != "n_episodes":
				self.logger.log_stat(
					prefix + k + "_mean", v / stats["n_episodes"], self.t_env
				)
		stats.clear()


def env_worker(remote, env_fn):
	# Make environment
	env = env_fn.x()
	while True:
		cmd, data = remote.recv()
		if cmd == "step":
			actions = data
			# Take a step in the environment
			_, reward, terminated, truncated, env_info = env.step(actions)
			terminated = terminated or truncated
			# Return the observations, avail_actions and state to make the next action
			state = env.get_state()
			avail_actions = env.get_avail_actions()
			obs = env.get_obs()
			remote.send({
				# Data for the next timestep needed to pick an action
				"state": state,
				"avail_actions": avail_actions,
				"obs": obs,
				# Rest of the data for the current timestep
				"reward": reward,
				"terminated": terminated,
				"info": env_info,
			})
		elif cmd == "reset":
			env.reset()
			remote.send({
				"state": env.get_state(),
				"avail_actions": env.get_avail_actions(),
				"obs": env.get_obs(),
			})
		elif cmd == "close":
			env.close()
			remote.close()
			break
		elif cmd == "get_env_info":
			remote.send(env.get_env_info())
		elif cmd == "get_stats":
			remote.send(env.get_stats())
		elif cmd == "render":
			env.render()
		elif cmd == "save_replay":
			env.save_replay()
		else:
			raise NotImplementedError


class CloudpickleWrapper:
	"""
	Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
	"""

	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle

		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle

		self.x = pickle.loads(ob)


def activation_rate(
	data,
	original_obs_shape,
	theory_index,
	env_theories,
	current_env_tested,
	load_step,
	env_obs_to_atom_fn,
	theory_dicts,
	env_action_map,
	agent_idx,
	path_to_save,
):
	activation_rate = compute_activation_rate(
		data,
		original_obs_shape,
		obs_to_atoms_fn=env_obs_to_atom_fn,
		theory=theory_dicts,
		action_map=env_action_map,
		agent_idx=agent_idx,
	)

	df = pd.DataFrame(
		[
			(
				theory_index,
				env_theories,
				current_env_tested,
				load_step,
				agent_idx,
				env_action_map[i],
				activation_rate[i],
			)
			for i in range(activation_rate.shape[0])
		],
		columns=[
			"seed",
			"env_theory",
			"env_test",
			"steps",
			"agent",
			"action",
			"activation_rate",
		],
	)

	df.to_csv(path_to_save, mode="a", header=False)


def compute_activation_rate(
	batch,
	original_obs_shape,
	obs_to_atoms_fn,
	theory,
	action_map,
	agent_idx=0,
):
	"""
	Args:
		batch (np.ndarray): batch matrix whose matrixes have shape n_episodes x n_steps x n_agents x action_size / flattened_obs_size
		original_obs_shape (tuple): original shape of the observation. We will reshape the last dimension of obs
									 (flattened_obs_size) to this
		obs_to_atoms_fn: function that takes an observation and returns the body atoms
		theory (dict): theory, in the form of a dictionary like {action: list(set(str))} where for each action there is a
						 list of rules expressed as a set of str (which are the atoms in the body of the rule)
		path (str, optional): path where to save the results.

	"""
	obs = batch["obs"]
	obs = obs.reshape(*obs.shape[:-1], *original_obs_shape).cpu().numpy()
	# Shape [batch, steps, n_agents, action.shape]
	actions = batch["actions"].cpu().numpy()

	actions_tot = np.zeros(len(action_map.keys()))
	actions_activation = np.zeros(len(action_map.keys()))

	for episode_idx in range(obs.shape[0]):
		for step_idx in range(obs.shape[1]):

			action = actions[episode_idx, step_idx, agent_idx].item()

			# We skip the action '0' which is not interesting
			if action == 0:
				continue
			obs_ = obs[episode_idx, step_idx, agent_idx]
			atoms = set(obs_to_atoms_fn(obs=obs_))

			# Increment the counter for possibly seen rules
			actions_tot[action] += 1

			# Check if the atoms activate one or more rules in the theory
			n_hits = 0
			for rule in theory[agent_idx][action_map[action]]:
				if _does_rule_activate(atoms, rule):
					n_hits = 1
			actions_activation[action] += n_hits

	res = np.nan_to_num(actions_activation / actions_tot, nan=0.0)
	return res

def _does_rule_activate(atoms, rule):
	"""
	Args:
		atoms (set): set of str representing the atoms that are true in the observation
		rule (set): set of str representing the body of the rule

	Returns:
		bool: True if the rule activates (i.e., all atoms in the body of the rule are true given the atoms in the atoms set)
	"""
	# Example:
	# Atoms: {'goal(east, 1)', 'ego(n)', 'goal(east, 0)', 'goal(north, 3)'}
	# Rule: {'goal(south, V1)', 'V1 <= 2', 'ego(n)'}

	# Separate rule elements
	# simple atoms is stuff like "ego(n)"
	simple_atoms = set()
	# predicates with variables is stuff like "goal(south, V1)"
	predicates_with_vars = []
	# constraints are expressions like "V1 <= 5"
	constraints = []
	
	for element in rule:
		if any(op in element for op in ['<=', '>=', '<', '>', '=']):
			constraints.append(element.strip())
		elif 'V' in element and '(' in element:
			predicates_with_vars.append(element.strip())
		else:
			simple_atoms.add(element.strip())
	
	# Check simple atoms first
	if not simple_atoms.issubset(atoms):
		return False
	
	# Match predicates with variables
	variable_bindings = {}
	
	for predicate in predicates_with_vars:
		# Parse: "goal(south, V1)" -> "goal", ["south", "V1"]
		pred_name = predicate.split('(')[0] # "goal"
		pred_args_str = predicate.split('(')[1].rstrip(')')
		pred_args = [arg.strip() for arg in pred_args_str.split(',')] # ["south", "V1"]
		
		found_match = False
		for atom in atoms:
			# we only care about atoms that are predicates
			if '(' not in atom:
				continue
				
			atom_name = atom.split('(')[0]
			atom_args_str = atom.split('(')[1].rstrip(')')
			atom_args = [arg.strip() for arg in atom_args_str.split(',')]
			
			# Check if names and arg counts match
			if pred_name != atom_name or len(pred_args) != len(atom_args):
				continue
			
			# Try to bind variables
			temp_bindings = {}
			all_match = True
			
			# check if each argument matches, considering variables 
			# ex: "goal(south, V1)" matches "goal(south, 3)" with binding V1=3
			for pred_arg, atom_arg in zip(pred_args, atom_args):
				if pred_arg.startswith('V'):
					# Variable
					if pred_arg in variable_bindings:
						if variable_bindings[pred_arg] != atom_arg:
							all_match = False
							break
					else:
						temp_bindings[pred_arg] = atom_arg
				else:
					# Constant  must match exactly
					if pred_arg != atom_arg:
						all_match = False
						break
			
			if all_match:
				variable_bindings.update(temp_bindings)
				found_match = True
				break
		
		if not found_match:
			return False
	
	# Evaluate constraints
	# so, if we have a constraint like "V1 <= 5" and we have a binding V1=3, we check if "3 <= 5" is true
	for constraint in constraints:
		try:
			eval_str = constraint
			for var, value in variable_bindings.items():
				try:
					numeric_value = float(value)
					eval_str = eval_str.replace(var, str(numeric_value))
				except ValueError:
					return False  # Can't evaluate constraint with non-numeric value
			
			if not eval(eval_str):
				return False
		except:
			return False
	
	return True