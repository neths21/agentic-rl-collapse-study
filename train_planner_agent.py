import gymnasium as gym
import minigrid

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from planner import generate_plan
from subtask_wrapper import SubtaskRewardWrapper


def main():

    plan = generate_plan()

    env = gym.make("MiniGrid-DoorKey-8x8-v0")

    env = ImgObsWrapper(env)

    env = SubtaskRewardWrapper(env)

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

    model.save("ppo_planner")

    print("Experiment 3 complete")


if __name__ == "__main__":
    main()