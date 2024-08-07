from stable_baselines3 import DQN  # pip install stable-baselines3

from luxai2021.env.agent import AgentFromStdInOut
from examples.rba_agent_v1 import RuleBasedAgent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
<<<<<<< HEAD:kaggle_submissions/PPO.py
from v4 import AgentPolicy
=======
from examples.agent_policy_DQN_1 import DQN_1_AgentPolicy
>>>>>>> origin/ahiyer-2:kaggle_submissions/main_lux-ai-2021.py

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
<<<<<<< HEAD:kaggle_submissions/PPO.py
    #model_id = 5403
    #total_steps = int(48e6)
    #model = PPO.load(f"models/rl_model_{model_id}_{total_steps}_steps.zip")
    model = PPO.load(f"./v8.zip")
    
=======
    model_id = 7113
    total_steps = int(900000)
    model = DQN.load(f"model7113_step1500000.zip")

>>>>>>> origin/ahiyer-2:kaggle_submissions/main_lux-ai-2021.py
    # Create a kaggle-remote opponent agent
    opponent = RuleBasedAgent()

    # Create a RL agent in inference mode
<<<<<<< HEAD:kaggle_submissions/PPO.py
    player = AgentPolicy(mode="inference", model=model)
    
=======
    player = DQN_1_AgentPolicy(mode="inference", model=model)

>>>>>>> origin/ahiyer-2:kaggle_submissions/main_lux-ai-2021.py
    # Run the environment
    env = LuxEnvironment(configs, player, opponent)
    env.reset()  # This will automatically run the game since there is
    # no controlling learning agent.
