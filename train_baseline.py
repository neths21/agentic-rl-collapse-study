import gymnasium as gym
import minigrid

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Create environment
env = gym.make("MiniGrid-DoorKey-8x8-v0")

# Convert observation to image only
env = ImgObsWrapper(env)

# Monitor wrapper
env = Monitor(env)

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/"
)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("ppo_doorkey")

print("Training finished.")