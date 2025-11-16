import os
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from SR_DQN import SR_DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class PlotRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PlotRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        # Sum rewards per episode
        self.current_rewards += self.locals['rewards']
        if self.locals['dones']:  # End of episode
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0  # Reset for next episode
        return True



def train_doorkey_env(env_id, model_type, timesteps):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env, filter_keys=['image']))
    ModelClass = SR_DQN if model_type == "SR_DQN" else DQN
    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "gamma": 0.95,
        "learning_starts": 1000
    }
    if model_type == "SR_DQN":
        model_kwargs.update({       # You can use these parameters to change the impact of the rules
            "conf_level": 0.8,              
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.05,
        })
    model = ModelClass(**model_kwargs)

    # Instantiate the custom callback
    reward_callback = PlotRewardCallback()

    # Train the model with the callback
    model.learn(
        total_timesteps=int(timesteps),
        progress_bar=True,
        callback=reward_callback            # Hint: you can slightly change the code to add the wandb callback and visualize training
    )

    return reward_callback.episode_rewards

def main():
    parser = argparse.ArgumentParser(description="Choose which environment and model to test")
    parser.add_argument('--map', 
                        type=str, 
                        choices=['5x5', '6x6', '8x8'], 
                        required=True, 
                        help="The map size to test: '5x5', '6x6' or '8x8'")
    parser.add_argument('--steps', 
                        type=int, 
                        required=True, 
                        help="How many training steps to perform")
    args = parser.parse_args()

    env_id = f"MiniGrid-DoorKey-{args.map}-v0"

    # Train both models
    rewards_dqn = train_doorkey_env(env_id=env_id, model_type="DQN", timesteps=args.steps)
    rewards_sr_dqn = train_doorkey_env(env_id=env_id, model_type="SR_DQN", timesteps=args.steps)

    # Plot the rewards comparison
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_dqn, label="DQN", alpha=0.7)
    plt.plot(rewards_sr_dqn, label="SR_DQN", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Comparison Between DQN and SR_DQN")
    plt.legend()
    plt.grid()
    plt.savefig("rewards.png")

if __name__ == "__main__":
    main()
