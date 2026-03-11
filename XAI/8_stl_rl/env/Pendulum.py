__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from env.rtamt_yml_parser import RTAMTYmlParser


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class Pendulum(gym.Env):
	"""
	## Description

	The inverted pendulum swingup problem is based on the classic problem in control theory.
	The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
	The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
	into an upright position, with its center of gravity right above the fixed point.

	The diagram below specifies the coordinate system used for the implementation of the pendulum's
	dynamic equations.

	![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

	- `x-y`: cartesian coordinates of the pendulum's end in meters.
	- `theta` : angle in radians.
	- `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

	## Action Space

	The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

	| Num | Action | Min  | Max |
	|-----|--------|------|-----|
	| 0   | Torque | -2.0 | 2.0 |

	## Observation Space

	The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
	end and its angular velocity.

	| Num | Observation      | Min  | Max |
	|-----|------------------|------|-----|
	| 0   | x = cos(theta)   | -1.0 | 1.0 |
	| 1   | y = sin(theta)   | -1.0 | 1.0 |
	| 2   | Angular Velocity | -8.0 | 8.0 |

	## Rewards

	The reward function is defined as:

	*r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

	where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
	Based on the above equation, the minimum reward that can be obtained is
	*-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
	while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

	## Starting State

	The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

	## Episode Truncation

	The episode truncates at 200 time steps.

	## Arguments

	- `g`: .

	Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
	the acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
	The default value is `g = 10.0`.
	On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

	```python
	>>> import gymnasium as gym
	>>> env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0
	>>> env
	<TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
	>>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5
	(array([ 0.4123625 ,  0.91101986, -0.89235795], dtype=float32), {})

	```

	## Version History

	* v1: Simplify the math equations, no difference in behavior.
	* v0: Initial versions release
	"""

	metadata = {
		"render_modes": ["human", "rgb_array"],
		"render_fps": 30,
	}

	def __init__(self, render_mode: Optional[str] = None, g=10.0):
		self.max_speed = 8
		self.max_torque = 2.0
		self.dt = 0.05
		self.g = g
		self.m = 1.0
		self.l = 1.0

		self.render_mode = render_mode

		self.screen_dim = 500
		self.screen = None
		self.clock = None
		self.isopen = True

		# 3 + 2 indicator variable
		high = np.array([1.0, 1.0, self.max_speed, 1.0, 1.0], dtype=np.float32)
		self.action_space = spaces.Box(
			low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
		)
		self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

		self.beta = 100.0
		self.pure_state_length = 3
		self.tau = 10
		self.n_subformulas = 1
		self.past_tau_trajectory = []
		self.parser = RTAMTYmlParser('env/pendulum_eventually_costs.yml')
		
		self.torque_constr = 0.3   # definition of constraint
		self.thetadot_constr = 0.5 # definition of constraint

	def step(self, u):
		th, thdot = self.state  # th := theta

		g = self.g
		m = self.m
		l = self.l
		dt = self.dt
		
		# theta: angolo attuale del pendolo rispetto alla posizione verticale
		# thetadot: velocità angolare
		# newtheta: nuova posizione angolare dopo l'applicazione della dinamica del pendolo
		# newthetadot: nuova velocità angolare dopo l'applicazione della dinamica del pendolo

		# normalize the angle th, clip the torque u
		th = angle_normalize(th)
		u = np.clip(u, -self.max_torque, self.max_torque)[0]
		
		# use self.parser.compute_robustness_dense to compute robustness over an horizon (self.past_tau_trajectory)
		total_rho, simple_rho = self.parser.compute_robustness_dense(self.past_tau_trajectory)
		# simple_tho: dictionary with individual robustness values for each specification
		
		# compute rho_tetadot_constraint and rho_torque_constraint, as separate variables
		rho_thetadot = simple_rho['thetadot_constraint']
		rho_torque = simple_rho['torque_constraint']  
		
		# apply tanh to the robustness to compute the smoother costs
		# critic cost
		# beta: scaling factor for the cost, higher beta means sharper transition between low and high cost
		# tanh: make function diff
		# costs: how much penalize the agent for violating the constraints
		cost_thetadot = np.tanh(self.beta * rho_thetadot)
		cost_torque = np.tanh(self.beta * rho_torque)
		
		# update thdot -> newthdot and th -> newth according to pendulum equations
		newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l**2) * u) * dt
		newth = th + newthdot * dt # normalization?
		newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

		simple_cost = 0.5 - abs(th)
		
		# reward defined in gymnasium: r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2)
		reward = -(angle_normalize(newth)**2 + 0.1 * newthdot**2 + 0.001 * (u**2))
		self.state = np.array([newth, newthdot])
		
		if self.render_mode == "human":
			self.render()

		return self._get_obs(np.array([newth, newthdot]), u), reward, False, False, {
			"cost_thetadot": cost_thetadot,
			"cost_torque": cost_torque,
			"rho_thetadot": rho_thetadot,
			"rho_torque": rho_torque,
			"no_stl_cost": simple_cost,
			"theta": angle_normalize(th),
			"thetadot": thdot,
			"torque": u
		}

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		super().reset(seed=seed)
		
		self.past_tau_trajectory = []
		
		if options is None:
			high = np.array([DEFAULT_X, DEFAULT_Y])
		else:
			# Note that if you use custom reset bounds, it may lead to out-of-bound
			# state/observations.
			x = options.get("x_init") if "x_init" in options else DEFAULT_X
			y = options.get("y_init") if "y_init" in options else DEFAULT_Y
			x = utils.verify_number_and_cast(x)
			y = utils.verify_number_and_cast(y)
			high = np.array([x, y])
		low = -high  # We enforce symmetric limits.
		self.state = self.np_random.uniform(low=low, high=high)
		self.last_u = None

		if self.render_mode == "human":
			self.render()
		return self._get_obs(self.state, 0), {}

	def _get_obs(self, next_state, action):
		"""
		Build the observation vector to return to the RL agent.
		The observation includes physical state + constraint satisfaction metrics.
		
		Returns a 5-dimensional observation:
			[cos(theta), sin(theta), thetadot, thetadot_robustness, torque_robustness]
		"""
		
		# ========== PART 1: Extract current state ==========
		# Get the current pendulum angle and angular velocity
		theta, thetadot = self.state
		observation = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
		
		# ========== PART 2: Maintain rolling history (10 time steps) ==========
		# This trajectory history is used to compute constraint satisfaction over a temporal window
		if len(self.past_tau_trajectory) == 0:
			# First call: initialize with placeholder entries
			# Each entry is [angular_velocity, torque] for each time step
			for i in range(self.tau):  # self.tau = 10
				self.past_tau_trajectory.append([thetadot, self.torque_constr + 1e-3])
		else:
			# Subsequent calls: maintain a sliding window of 10 most recent steps
			# Remove the oldest entry (shift left) and append the newest entry
			self.past_tau_trajectory = self.past_tau_trajectory[1:]  # Remove first element
			self.past_tau_trajectory.append([thetadot, self.last_u])  # Add current [velocity, action]
				
		tau_num = len(self.past_tau_trajectory)
		assert tau_num == self.tau, "dim of tau-state is wrong."
		
		# ========== PART 3: Compute constraint robustness over the horizon ==========
		# These are STL "Globally" constraints: they must be satisfied at EVERY time step
		# If ANY time step violates the constraint, robustness resets to 0
		
		thetadot_rho = 0  # Robustness for |angular_velocity| <= 0.5
		torque_rho = 0    # Robustness for |torque| <= 0.3
		
		for i in range(tau_num):
			# ===== Angular velocity constraint =====
			# Check if |thetadot| <= self.thetadot_constr (0.5)
			if (self.thetadot_constr - abs(self.past_tau_trajectory[i][0])) >= 0:
				# Constraint satisfied at this time step: accumulate robustness
				thetadot_rho = min(thetadot_rho + 1 / (float(self.tau + 1)), 1.0)
			else:
				# Constraint violated at ANY time step: set robustness to 0 (hard reset)
				thetadot_rho = 0.0
			
			# ===== Torque constraint =====
			# Check if |torque| <= self.torque_constr (0.3)
			if (self.torque_constr - abs(self.past_tau_trajectory[i][1])) >= 0:
				# Constraint satisfied at this time step: accumulate robustness
				torque_rho = min(torque_rho + 1 / (float(self.tau + 1)), 1.0)
			else:
				# Constraint violated at ANY time step: set robustness to 0 (hard reset)
				torque_rho = 0.0
		
		# ========== PART 4: Normalize robustness values ==========
		# Shift from [0, 1] range to [-0.5, 0.5] for better RL learning dynamics
		# Negative values = constraints violated; Positive values = constraints satisfied
		thetadot_rho -= 0.5  # Range: [-0.5, 0.5]
		torque_rho -= 0.5    # Range: [-0.5, 0.5]
		
		# ========== PART 5: Build final observation and save state ==========
		# Save action and state for next step's trajectory computation
		self.last_u = action      # Save action for use in next _get_obs call
		self.state = next_state   # Update internal state
		next_theta, next_thetadot = self.state
		
		# Construct final 5-dimensional observation for the agent
		observation = np.array(
			[np.cos(next_theta), np.sin(next_theta), next_thetadot, thetadot_rho, torque_rho],
			dtype=np.float32
		)
		
		return observation

	def render(self):
		if self.render_mode is None:
			assert self.spec is not None
			gym.logger.warn(
				"You are calling render method without specifying any render mode. "
				"You can specify the render_mode at initialization, "
				f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
			)
			return

		try:
			import pygame
			from pygame import gfxdraw
		except ImportError as e:
			raise DependencyNotInstalled(
				'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
			) from e

		if self.screen is None:
			pygame.init()
			if self.render_mode == "human":
				pygame.display.init()
				self.screen = pygame.display.set_mode(
					(self.screen_dim, self.screen_dim)
				)
			else:  # mode in "rgb_array"
				self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
		if self.clock is None:
			self.clock = pygame.time.Clock()

		self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
		self.surf.fill((255, 255, 255))

		bound = 2.2
		scale = self.screen_dim / (bound * 2)
		offset = self.screen_dim // 2

		rod_length = 1 * scale
		rod_width = 0.2 * scale
		l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
		coords = [(l, b), (l, t), (r, t), (r, b)]
		transformed_coords = []
		for c in coords:
			c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
			c = (c[0] + offset, c[1] + offset)
			transformed_coords.append(c)
		gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
		gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

		gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
		gfxdraw.filled_circle(
			self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
		)

		rod_end = (rod_length, 0)
		rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
		rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
		gfxdraw.aacircle(
			self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
		)
		gfxdraw.filled_circle(
			self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
		)

		fname = path.join(path.dirname(__file__), "assets/clockwise.png")
		img = pygame.image.load(fname)
		if self.last_u is not None:
			scale_img = pygame.transform.smoothscale(
				img,
				(
					float(scale * np.abs(self.last_u) / 2),
					float(scale * np.abs(self.last_u) / 2),
				),
			)
			is_flip = bool(self.last_u > 0)
			scale_img = pygame.transform.flip(scale_img, is_flip, True)
			self.surf.blit(
				scale_img,
				(
					offset - scale_img.get_rect().centerx,
					offset - scale_img.get_rect().centery,
				),
			)

		# drawing axle
		gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
		gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

		self.surf = pygame.transform.flip(self.surf, False, True)
		self.screen.blit(self.surf, (0, 0))
		if self.render_mode == "human":
			pygame.event.pump()
			self.clock.tick(self.metadata["render_fps"])
			pygame.display.flip()

		else:  # mode == "rgb_array":
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
			)

	def close(self):
		if self.screen is not None:
			import pygame

			pygame.display.quit()
			pygame.quit()
			self.isopen = False


def angle_normalize(x):
	return ((x + np.pi) % (2 * np.pi)) - np.pi
