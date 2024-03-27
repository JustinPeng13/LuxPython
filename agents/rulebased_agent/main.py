import random
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

from agent import RuleBasedAgent

if __name__ == "__main__":
    # Create a game environment
    configs = LuxMatchConfigs_Default
    configs["seed"] = random.randint(-1000000, 1000000)
    env = LuxEnvironment(configs, RuleBasedAgent(), RuleBasedAgent())

    env.game.start_replay_logging()
    is_game_error = env.run_no_learn()

    if not is_game_error:
        print(f"Game done turn {env.game.state['turn']}, final map:")
        print(env.game.map.get_map_string())
    else:
        raise Exception("Game error")
