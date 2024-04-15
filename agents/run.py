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

def get_command_line_arguments():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Training script for Lux RL agent.')
    parser.add_argument('--id', help='Identifier of this run', type=str, default=str(random.randint(0, 10000)))
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.0001)
    parser.add_argument('--gamma', help='Gamma', type=float, default=0.999)
    parser.add_argument('--gae_lambda', help='GAE Lambda', type=float, default=0.95)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=128)  # 64
    parser.add_argument('--step_count', help='Total number of steps to train', type=int, default=100000000000000000)
    parser.add_argument('--n_steps', help='Number of experiences to gather before each learning period', type=int, default=1024)
    parser.add_argument('--path', help='Path to a checkpoint to load to resume training', type=str, default=None)
    parser.add_argument('--n_envs', help='Number of parallel environments to use in training', type=int, default=1)
    parser.add_argument('--device', help='Device to use in training', type=str, default="cuda")
    args = parser.parse_args()

    return args


def run_matches(args, num_matches=10):
    print(args)

    ALGO = PPO
    latest_save = "./models/model6203_step5750000.zip"

    model = ALGO.load(latest_save)
    model.lr_schedule = get_schedule_fn(args.learning_rate)

    configs = LuxMatchConfigs_Default
    player = RLAgent(mode="inference", model=model)
    opponent = RuleBasedAgent()

    for i in range(num_matches): # run 5 matches
        env = LuxEnvironment(configs=configs,
                         learning_agent=player,
                         opponent_agent=opponent
        )
        env.game.configs["seed"] = random.randint(-10000,10000)
        env.game.start_replay_logging(stateful=True, replay_folder="./replays/", replay_filename_prefix="replay")
        model.set_env(env)
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

    # Get the command line arguments
    local_args = get_command_line_arguments()

    # Run the match
    run_matches(local_args)
