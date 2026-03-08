import gymnasium as gym
import minigrid

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

# Create environment
env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")

# IMPORTANT: same wrapper used during training
env = ImgObsWrapper(env)

# Load trained model
model = PPO.load("ppo_llama3_reward")

obs, info = env.reset()

for _ in range(500):

    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()