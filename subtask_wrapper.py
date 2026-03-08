import gymnasium as gym
import numpy as np

class SubtaskRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.visited = set()

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)
        self.visited = set()

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        # agent position
        pos = tuple(self.env.unwrapped.agent_pos)

        # reward exploration
        if pos not in self.visited:
            reward += 0.05
            self.visited.add(pos)

        return obs, reward, terminated, truncated, info