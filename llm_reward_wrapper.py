from gymnasium import Wrapper
from critic import llm_reward


class LLMRewardWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.prev_state = None
        self.step_count = 0
        self.llm_called = False

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        self.prev_state = obs
        self.step_count = 0
        self.llm_called = False

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1

        # Call LLM once per episode
        if reward == 0 and not self.llm_called and self.step_count > 50:

            llm_r = llm_reward(self.prev_state, action, obs)

            reward += 0.05 * llm_r

            self.llm_called = True

        self.prev_state = obs

        return obs, reward, terminated, truncated, info