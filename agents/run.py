import argparse
import os
import sys
import random

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import get_schedule_fn

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

from rl_agent.agent import RLAgent
from rulebased_agent.agent import RuleBasedAgent

def run_matches(num_matches=50):
    configs = LuxMatchConfigs_Default
    player = RLAgent(mode="inference", model=PPO.load("./models/model3878_step10700000.zip"))
    opponent = RuleBasedAgent()

    for i in range(num_matches):
        configs["seed"] = random.randint(-10000, 10000)
        env = LuxEnvironment(configs=configs,
                         learning_agent=player,
                         opponent_agent=opponent
        )
        env.game.start_replay_logging(stateful=True, replay_folder="./replays/", replay_filename_prefix="replay")
        env.run_no_learn()
        env.close()

if __name__ == "__main__":
    if sys.version_info < (3,7) or sys.version_info >= (3,8):
        os.system("")
        class style():
            YELLOW = '\033[93m'
        version = str(sys.version_info.major) + "." + str(sys.version_info.minor)
        message = f'/!\ Warning, python{version} detected, you will need to use python3.7 to submit to kaggle.'
        message = style.YELLOW + message
        print(message)

    run_matches()
