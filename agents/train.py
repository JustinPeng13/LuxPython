import argparse
import glob
import os
import sys
import random

from stable_baselines3 import PPO, DQN  # pip install stable-baselines3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default

from rl_agent.agent import RLAgent
from rulebased_agent.agent import RuleBasedAgent

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment 
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


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


def train(args):
    """
    The main training loop
    :param args: (ArgumentParser) The command line arguments
    """
    print(args)

    ALGO = PPO
    algo_policy = "MlpPolicy"

    # Run a training job
    configs = LuxMatchConfigs_Default

    # Train against dummy agent or rule based agent
    # opponent = Agent()
    opponent = RuleBasedAgent()

    # Create a RL agent in training mode
    player = RLAgent(mode="train")

    # Train the model
    env_eval = None
    if args.n_envs == 1:
        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)
    else:
        env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=RLAgent(mode="train"),
                                                     opponent_agent=RuleBasedAgent()), i) for i in range(args.n_envs)])
    
    run_id = args.id
    print("Run id %s" % run_id)

    if args.path:
        print('using previous args', args, args.path)
        # by default previous model params are used (lr, batch size, gamma...)
        model = ALGO.load(args.path)
        model.set_env(env=env)

        # Update the learning rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)

        # TODO: Update other training parameters
    else:
        print('using new args', args)
        model = ALGO(
            algo_policy,
            env,
            verbose=1,
            tensorboard_log="./lux_tensorboard/",
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            device=args.device
        )

    # Save a checkpoint and 5 match replay files every 100K steps
    callbacks = [
        SaveReplayAndModelCallback(
            save_freq=50000,
            save_path='./models/',
            name_prefix=f'model{run_id}',
            replay_env=LuxEnvironment(
                configs=configs,
                learning_agent=RLAgent(mode="inference", model=model),
                opponent_agent=RLAgent(mode="inference", model=model)
            ),
            replay_num_episodes=5
        )
    ]

    print("Training model...")
    model.learn(total_timesteps=args.step_count, callback=callbacks)
    if not os.path.exists(f'models/rl_model_{run_id}_{args.step_count}_steps.zip'):
        model.save(path=f'models/rl_model_{run_id}_{args.step_count}_steps.zip')
    print("Done training model.")

    # Inference the model and print to console
    # print("Inference model policy with rendering...")
    # saves = glob.glob(f'models/rl_model_{run_id}_*_steps.zip')
    # latest_save = sorted(saves, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
    # model.load(path=latest_save)
    # obs = env.reset()
    # for i in range(600):
    #     action_code, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, done, info = env.step(action_code)
    #     if i % 5 == 0:
    #         print("Turn %i" % i)
    #         env.render()

    #     if done:
    #         print("Episode done, resetting.")
    #         obs = env.reset()
    # print("Done")

    
    # Learn with self-play against the learned model as an opponent now
    print("Training model with self-play against last version of model...")
    # player = RLAgent(mode="train")
    opponent = RLAgent(mode="inference", model=model)
    env = LuxEnvironment(configs, player, opponent)
    model = ALGO(
        algo_policy,
        env,
        verbose=1,
        tensorboard_log="./lux_tensorboard/",
        learning_rate = 0.0003,
        gamma=0.999,
        gae_lambda = 0.95,
        device=args.device
    )
    model.learn(total_timesteps=50000, callback=callbacks)

    env.close()
    print("Done")


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

    # Train the model
    train(local_args)
