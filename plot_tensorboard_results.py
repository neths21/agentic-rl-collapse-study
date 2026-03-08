import pandas as pd
import matplotlib.pyplot as plt

# Episode Length
length = pd.read_csv("results/PPO_1.csv")

plt.figure(figsize=(8,5))
plt.plot(length["Step"], length["Value"])
plt.xlabel("Training Timesteps")
plt.ylabel("Episode Length")
plt.title("Episode Length vs Training Steps")
plt.grid(True)
plt.tight_layout()
plt.savefig("episode_length_curve.png", dpi=300)

# Episode Reward
reward = pd.read_csv("results/PPO_2.csv")

plt.figure(figsize=(8,5))
plt.plot(reward["Step"], reward["Value"])
plt.xlabel("Training Timesteps")
plt.ylabel("Episode Reward")
plt.title("Reward vs Training Steps")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=300)

plt.show()