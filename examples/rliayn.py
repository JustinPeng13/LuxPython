import sys
import time
from functools import partial  # pip install functools
import copy
import random

import numpy as np
from gym import spaces

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)


def smart_transfer_to_nearby(
    game, team, unit_id, unit, target_type_restriction=None, **kwarg
):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if (
                                    u.get_cargo_space_left() >= resource_amount
                                    and target_unit.get_cargo_space_left()
                                    >= resource_amount
                                ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if (
                                        u.get_cargo_space_left()
                                        < target_unit.get_cargo_space_left()
                                    ):
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u

                                elif (
                                    target_unit.get_cargo_space_left()
                                    >= resource_amount
                                ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass

                                elif (
                                    u.get_cargo_space_left()
                                    > target_unit.get_cargo_space_left()
                                ):
                                    # Change targets, because neither target can accept all our resources and
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u

    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()

    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)


# ########################################################################################################################
# # This is the Agent that you need to design for the competition
# ########################################################################################################################
class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(
                MoveAction, direction=Constants.DIRECTIONS.CENTER
            ),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            # partial(
            #     smart_transfer_to_nearby,
            #     target_type_restriction=Constants.UNIT_TYPES.CART,
            # ),  # Transfer to nearby cart
            # partial(
            #     smart_transfer_to_nearby,
            #     target_type_restriction=Constants.UNIT_TYPES.WORKER,
            # ),  # Transfer to nearby worker
            SpawnCityAction,
            # PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            # SpawnCartAction,
            # ResearchAction,
        ]
        self.action_space = spaces.Discrete(
            max(len(self.actions_units), len(self.actions_cities))
        )

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is citytile
        #
        #   1x cooldown %
        #   1x wood cargo %
        #   1x coal cargo %
        #   1x uranium cargo %
        #   1x cargo is full
        #   1x can build city
        #   1x will survive night
        #   1x will survive game
        #
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_dying_city
        #   1x distance_nearest_dying_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        # State:
        #   1x is night
        #   1x turn in day/night cycle
        #   1x percent of game done
        #   1x citytile counts [cur player]
        #   1x worker counts [cur player]
        #   1x research points [cur player]
        #   1x can farm coal [cur player]
        #   1x can farm uranium [cur player]
        # Map:
        #   1x % wood left
        #   1x % coal left
        #   1x % uranium left
        self.observation_shape = (2 + 8 + 7 * 6 + 8 + 3,)  # total 63
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.observation_shape, dtype=np.float32
        )

        self.object_nodes = {}
        self.total_resources = {
            Constants.RESOURCE_TYPES.WOOD: 0,
            Constants.RESOURCE_TYPES.COAL: 0,
            Constants.RESOURCE_TYPES.URANIUM: 0,
        }

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        if is_new_turn:
            # It's a new turn this event. This flag is set True for only the first observation from each turn.
            # Update any per-turn fixed observation space that doesn't change per unit/city controlled.

            # Build a list of object nodes by type for quick distance-searches
            self.object_nodes = {}

            # Add resources
            self.curr_resources = {
                Constants.RESOURCE_TYPES.WOOD: 0,
                Constants.RESOURCE_TYPES.COAL: 0,
                Constants.RESOURCE_TYPES.URANIUM: 0,
            }
            for cell in game.map.resources:
                self.object_nodes[cell.resource.type] = np.concatenate(
                    (
                        self.object_nodes.get(
                            cell.resource.type, np.empty((0, 2), dtype=int)
                        ),
                        np.array([[cell.pos.x, cell.pos.y]]),
                    ),
                    axis=0,
                )
                self.curr_resources[cell.resource.type] += cell.resource.amount

            # Add your own units
            for u in game.state["teamStates"][team]["units"].values():
                key = str(u.type)

                self.object_nodes[key] = np.concatenate(
                    (
                        self.object_nodes.get(key, np.empty((0, 2), dtype=int)),
                        np.array([[u.pos.x, u.pos.y]]),
                    ),
                    axis=0,
                )

            # Add your own cities
            for city in game.cities.values():
                for cells in city.city_cells:
                    key = "city"
                    if city.team != team:
                        continue

                    fuel_needed_cycle = city.get_light_upkeep() * min(
                        10, 40 - game.state["turn"] % 40
                    )
                    fuel_needed_total = (
                        city.get_light_upkeep()
                        * (9 - game.state["turn"] // 40 + 1)
                        * 10
                        + fuel_needed_cycle
                    )
                    if city.fuel < fuel_needed_total:
                        self.object_nodes["dying_city"] = np.concatenate(
                            (
                                self.object_nodes.get(
                                    "dying_city", np.empty((0, 2), dtype=int)
                                ),
                                np.array([[cells.pos.x, cells.pos.y]]),
                            ),
                            axis=0,
                        )

                    self.object_nodes[key] = np.concatenate(
                        (
                            self.object_nodes.get(key, np.empty((0, 2), dtype=int)),
                            np.array([[cells.pos.x, cells.pos.y]]),
                        ),
                        axis=0,
                    )

        obs = np.zeros(self.observation_shape)

        # Unit/City:
        #   1x is worker
        #   1x is citytile
        observation_index = 0
        if unit and unit.is_worker():
            obs[observation_index] = 1.0
        elif city_tile:
            obs[observation_index + 1] = 1.0
        observation_index += 2

        #   1x cooldown %
        #   1x wood cargo %
        #   1x coal cargo %
        #   1x uranium cargo %
        #   1x cargo is full
        #   1x can build city
        #   1x will survive night
        #   1x will survive game
        if unit:
            pos = unit.pos

            obs[observation_index] = (
                unit.cooldown
                / GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
            )
            obs[observation_index + 1] = (
                unit.cargo[Constants.RESOURCE_TYPES.WOOD] / 100.0
            )
            obs[observation_index + 2] = (
                unit.cargo[Constants.RESOURCE_TYPES.COAL] / 100.0
            )
            obs[observation_index + 3] = (
                unit.cargo[Constants.RESOURCE_TYPES.URANIUM] / 100.0
            )
            if (
                obs[observation_index + 1] == 1.0
                or obs[observation_index + 2] == 1.0
                or obs[observation_index + 3] == 1.0
            ):
                obs[observation_index + 4] = 1.0

            obs[observation_index + 5] = 1.0 if unit.can_build(game.map) else 0.0

            fuel_needed_cycle = unit.get_light_upkeep() * min(
                10, 40 - game.state["turn"] % 40
            )
            obs[observation_index + 6] = (
                1.0 if unit.get_cargo_fuel_value() >= fuel_needed_cycle else 0.0
            )
            fuel_needed_total = (
                unit.get_light_upkeep() * (9 - game.state["turn"] // 40 + 1) * 10
                + fuel_needed_cycle
            )
            obs[observation_index + 7] = (
                1.0 if unit.get_cargo_fuel_value() >= fuel_needed_total else 0.0
            )

            observation_index += 8

        elif city_tile:
            pos = city_tile.pos

            obs[observation_index] = (
                city_tile.cooldown
                / GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
            )

            city = game.cities[city_tile.city_id]
            fuel_needed_cycle = city.get_light_upkeep() * min(
                10, 40 - game.state["turn"] % 40
            )
            obs[observation_index + 6] = 1.0 if city.fuel >= fuel_needed_cycle else 0.0
            fuel_needed_total = (
                city.get_light_upkeep() * (8 - game.state["turn"] // 40) * 10
                + fuel_needed_cycle
            )
            obs[observation_index + 7] = 1.0 if city.fuel >= fuel_needed_total else 0.0

            observation_index += 8

        else:
            observation_index += 50  # Skip the rest of the unit/city observations

        if pos:
            for key in [
                Constants.RESOURCE_TYPES.WOOD,
                Constants.RESOURCE_TYPES.COAL,
                Constants.RESOURCE_TYPES.URANIUM,
                "city",
                "dying_city",
                str(Constants.UNIT_TYPES.WORKER),
            ]:
                # Process the direction to and distance to this object type

                # Encode the direction to the nearest object (excluding itself)
                #   5x direction
                #   1x distance
                if key in self.object_nodes:
                    if ("city" in key and city_tile) or (
                        unit
                        and str(unit.type) == key
                        and len(game.map.get_cell_by_pos(unit.pos).units) <= 1
                    ):
                        # Filter out the current unit from the closest-search of same unit type
                        closest_index = closest_node(
                            (pos.x, pos.y), self.object_nodes[key]
                        )
                        filtered_nodes = np.delete(
                            self.object_nodes[key], closest_index, axis=0
                        )
                    else:
                        filtered_nodes = self.object_nodes[key]

                    if len(filtered_nodes) == 0:
                        # No other object of this type
                        obs[observation_index + 5] = 1.0  # Distance to nearest = max
                    else:
                        # There is another object of this type
                        closest_index = closest_node((pos.x, pos.y), filtered_nodes)

                        if closest_index and closest_index >= 0:
                            closest = filtered_nodes[closest_index]
                            closest_position = Position(closest[0], closest[1])
                            direction = pos.direction_to(closest_position)
                            mapping = {
                                Constants.DIRECTIONS.CENTER: 0,
                                Constants.DIRECTIONS.NORTH: 1,
                                Constants.DIRECTIONS.WEST: 2,
                                Constants.DIRECTIONS.SOUTH: 3,
                                Constants.DIRECTIONS.EAST: 4,
                            }
                            obs[observation_index + mapping[direction]] = (
                                1.0  # One-hot encoding direction
                            )

                            # 0 to 1 distance (max 20 tiles)
                            distance = pos.distance_to(closest_position)
                            obs[observation_index + 5] = min(distance / 20.0, 1.0)

                            # 0 to 1 value (amount of resource, cargo for unit, or fuel for city)
                            if key == "city":
                                # City fuel as % of upkeep for 200 turns
                                c = game.cities[
                                    game.map.get_cell_by_pos(
                                        closest_position
                                    ).city_tile.city_id
                                ]
                                obs[observation_index + 6] = min(
                                    c.fuel / (c.get_light_upkeep() * 200.0), 1.0
                                )
                            elif key in [
                                Constants.RESOURCE_TYPES.WOOD,
                                Constants.RESOURCE_TYPES.COAL,
                                Constants.RESOURCE_TYPES.URANIUM,
                            ]:
                                # Resource amount (max 500 units)
                                obs[observation_index + 6] = min(
                                    game.map.get_cell_by_pos(
                                        closest_position
                                    ).resource.amount
                                    / 500,
                                    1.0,
                                )
                            else:
                                # Unit cargo
                                obs[observation_index + 6] = min(
                                    next(
                                        iter(
                                            game.map.get_cell_by_pos(
                                                closest_position
                                            ).units.values()
                                        )
                                    ).get_cargo_space_left()
                                    / 100,
                                    1.0,
                                )

                observation_index += 7

        # State:
        #   1x is night
        #   1x turn in day/night cycle
        #   1x percent of game done
        #   1x citytile counts [cur player]
        #   1x worker counts [cur player]
        #   1x research points [cur player]
        #   1x can farm coal [cur player]
        #   1x can farm uranium [cur player]
        #   1x is night
        obs[observation_index] = game.is_night()
        obs[observation_index + 1] = game.state["turn"] % 40 / 39
        obs[observation_index + 2] = (
            game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        )

        max_count = 30
        obs[observation_index + 3] = len(self.object_nodes.get("city", [])) / max_count
        obs[observation_index + 4] = (
            len(self.object_nodes.get(str(Constants.UNIT_TYPES.WORKER), [])) / max_count
        )

        obs[observation_index + 5] = (
            game.state["teamStates"][team]["researchPoints"] / 200.0
        )
        obs[observation_index + 6] = float(
            game.state["teamStates"][team]["researched"]["coal"]
        )
        obs[observation_index + 7] = float(
            game.state["teamStates"][team]["researched"]["uranium"]
        )

        observation_index += 8

        # Map:
        #   1x % wood left
        #   1x % coal left
        #   1x % uranium left
        obs[observation_index] = (
            self.curr_resources[Constants.RESOURCE_TYPES.WOOD]
            / self.total_resources[Constants.RESOURCE_TYPES.WOOD]
        )
        obs[observation_index + 1] = (
            self.curr_resources[Constants.RESOURCE_TYPES.COAL]
            / self.total_resources[Constants.RESOURCE_TYPES.COAL]
        )
        obs[observation_index + 2] = (
            self.curr_resources[Constants.RESOURCE_TYPES.URANIUM]
            / self.total_resources[Constants.RESOURCE_TYPES.URANIUM]
        )

        return obs

    def action_code_to_action(
        self, action_code, game, unit=None, city_tile=None, team=None
    ):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y,
                )
            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y,
                )

            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.cities_last = 0
        self.fuel_collected_last = 0
        self.score_last = 0

        for cell in game.map.resources:
            self.total_resources[cell.resource.type] += cell.resource.amount
        self.curr_resources = self.total_resources.copy()

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -100

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_tile_count = 0
        city_survival_score = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
                fuel_needed_total = city.get_light_upkeep() * (
                    8 - game.state["turn"] // 40
                ) * 10 + city.get_light_upkeep() * min(10, 40 - game.state["turn"] % 40)
                if city.fuel >= fuel_needed_total:
                    city_survival_score += len(city.city_cells)

            for _ in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1

        rewards = {}

        farm_score = (
            game.stats["teamStats"][self.team]["resourcesCollected"][
                Constants.RESOURCE_TYPES.WOOD
            ]
            / self.total_resources[Constants.RESOURCE_TYPES.WOOD]
            * 0.6
            + game.stats["teamStats"][self.team]["resourcesCollected"][
                Constants.RESOURCE_TYPES.COAL
            ]
            / self.total_resources[Constants.RESOURCE_TYPES.COAL]
            * 0.3
            + game.stats["teamStats"][self.team]["resourcesCollected"][
                Constants.RESOURCE_TYPES.URANIUM
            ]
            / self.total_resources[Constants.RESOURCE_TYPES.URANIUM]
            * 0.1
        )

        # # Give a reward for unit creation/death. 0.05 reward per unit.
        # rewards["rew/r_units"] = (unit_count - self.units_last) * 0.075
        # self.units_last = unit_count

        # # Give a reward for city creation/death. 0.1 reward per city.
        # rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        # self.city_tiles_last = city_tile_count

        # # Penalty for separate cities
        # rewards["rew/r_cities"] = (self.cities_last - city_count) * 0.1
        # self.cities_last = city_count

        # # Reward collecting fuel
        # fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        # rewards["rew/r_fuel_collected"] = (
        #     fuel_collected - self.fuel_collected_last
        # ) / 20000
        # self.fuel_collected_last = fuel_collected

        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        rewards["rew/r_game_win"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count * 2

            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = (
                    10 * game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
                )  # Win
            else:
                rewards["rew/r_game_win"] = -10  # Loss

            if random.random() < 0.012:
                print(
                    f"Game win: {game.get_winning_team() == self.team}\n \
                    Number of rounds: {game.state['turn']}\n \
                    Number of units: {unit_count}\n \
                      Farming score: {farm_score * 10000}\n \
                        City survival score: {city_survival_score}\n \
                        City tiles: {city_tile_count} \
                        City tile score: {rewards['rew/r_city_tiles_end']}"
                )

        # reward = 0
        # for name, value in rewards.items():
        #     reward += value
        curr_score = (
            farm_score * 10000
            + city_survival_score
            + rewards["rew/r_game_win"]
            + rewards["rew/r_city_tiles_end"]
        )
        reward = curr_score - self.score_last
        self.score_last = curr_score

        # print("reward", reward)
        return reward

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return
