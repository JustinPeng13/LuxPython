from stable_baselines3 import DQN  # pip install stable-baselines3

from luxai2021.env.agent import AgentFromStdInOut
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from agent_policy_DQN_1 import DQN_1_AgentPolicy
from agent_policy_DQN_2 import DQN_2_AgentPolicy
from agent_policy_DQN_3 import DQN_3_AgentPolicy
from agent_policy_DQN_4 import DQN_4_AgentPolicy

if __name__ == "__main__":
    """
    This is a kaggle submission, so we don't use command-line args
    and assume the model is in model.zip in the current folder.
    """
    # Tool to run this against itself locally:
    # "lux-ai-2021 --seed=100 main_lux-ai-2021.py main_lux-ai-2021.py --maxtime 10000"

    # Run a kaggle submission with the specified model
    configs = LuxMatchConfigs_Default

    # Load the saved model
    model = DQN.load(f"DQN2_1o2m.zip")

    # Create a kaggle-remote opponent agent
    opponent = AgentFromStdInOut()

    # Create a RL agent in inference mode
    player = DQN_2_AgentPolicy(mode="inference", model=model)

    # Run the environment
    env = LuxEnvironment(configs, player, opponent)
    env.reset()  # This will automatically run the game since there is
    # no controlling learning agent.
