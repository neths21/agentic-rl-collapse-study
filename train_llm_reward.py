import gymnasium as gym
import minigrid

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from llm_reward_wrapper import LLMRewardWrapper


env = gym.make("MiniGrid-DoorKey-8x8-v0")

env = ImgObsWrapper(env)

env = LLMRewardWrapper(env)

env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=500000)

model.save("ppo_llama3_reward")

print("Experiment with LLaMA3 complete")