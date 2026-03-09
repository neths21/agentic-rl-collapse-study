import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

env = gym.make("MiniGrid-DoorKey-8x8-v0")
env = ImgObsWrapper(env)

model = PPO.load("ppo_adaptive")

episodes = 50

success = 0
total_rewards = []
episode_lengths = []

for ep in range(episodes):

    obs, info = env.reset()
    done = False
    ep_reward = 0
    steps = 0

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += reward
        steps += 1

        done = terminated or truncated

    if ep_reward > 0:
        success += 1

    total_rewards.append(ep_reward)
    episode_lengths.append(steps)

print("Success rate:", success / episodes)
print("Average reward:", sum(total_rewards)/episodes)
print("Average episode length:", sum(episode_lengths)/episodes)