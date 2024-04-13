from stable_baselines3 import DQN  # pip install stable-baselines3

from luxai2021.env.agent import AgentFromStdInOut
from examples.rba_agent_v1 import RuleBasedAgent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from examples.agent_policy_DQN_1 import DQN_1_AgentPolicy

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
    model_id = 7113
    total_steps = int(900000)
    model = DQN.load(f"../examples/models/model7113_step1500000.zip")
    
    # Create a kaggle-remote opponent agent
    opponent = RuleBasedAgent()

    # Create a RL agent in inference mode
    player = DQN_1_AgentPolicy(mode="inference", model=model)

    # Run the environment
    env = LuxEnvironment(configs, player, opponent)
    env.reset()  # This will automatically run the game since there is
    # no controlling learning agent.
