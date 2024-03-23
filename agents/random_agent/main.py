import random
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

from agent import RandomAgent

# if __name__ == "__main__":
#     random_agent = RandomAgent()
#     game = Game(configs=LuxMatchConfigs_Default, agents=[random_agent])
#     # game = Game(configs=LuxMatchConfigs_Default)
#     game.configs["seed"] = 0
#     game.start_replay_logging()

#     game_over = False
#     while not game_over:
#         print("Turn %i" % game.state["turn"])

#         # Array of actions for both teams. Eg: MoveAction(team, unit_id, direction)
#         actions = []
#         # actions = agent(observation, None) # configuration = None
#         game_over = game.run_turn_with_actions(actions)

#     print("Game done, final map:")
#     print(game.map.get_map_string())


if __name__ == "__main__":
    player = RandomAgent()
    opponent = RandomAgent()

    # Create a game environment
    configs = LuxMatchConfigs_Default
    configs["seed"] = random.randint(1, 100)
    env = LuxEnvironment(configs=configs,
                         learning_agent=player,
                         opponent_agent=opponent)

    env.game.start_replay_logging()
    is_game_error = env.run_no_learn()

    if not is_game_error:
        print(f"Game done turn {env.game.state['turn']}, final map:")
        print(env.game.map.get_map_string())
    else:
        raise Exception("Game error")
