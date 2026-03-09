import gymnasium as gym
from collapse_detector import CollapseDetector
from critic import llm_reward


class AdaptiveRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.detector = CollapseDetector()
        self.step_count = 0
        self.lambda_weight = 0.1
        self.visited = set()

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        self.step_count = 0
        self.visited = set()

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1

        # exploration reward
        pos = tuple(self.env.unwrapped.agent_pos)

        if pos not in self.visited:
            reward += 0.01
            self.visited.add(pos)

        # collapse detection
        collapse = self.detector.update(action)

        if collapse:
            self.lambda_weight = 0.05
        else:
            self.lambda_weight = 0.1

        # LLM reward (rarely called)
        llm_r = 0

        if self.step_count > 5000 and self.step_count % 200 == 0:
            llm_r = llm_reward(action)

        reward = reward + self.lambda_weight * llm_r

        # small step penalty
        reward -= 0.002

        return obs, reward, terminated, truncated, info