import random
from functools import partial
from luxai2021.env.agent import Agent
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS

def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
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
                                if( u.get_cargo_space_left() >= resource_amount and 
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u
                                    
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass
                                    
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
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

########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class RuleBasedAgent(Agent):
    def __init__(self) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__()
        self.move_action_makers = {
            'n' : partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            's' : partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            'e' : partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            'w' : partial(MoveAction, direction=Constants.DIRECTIONS.WEST)
        }
        self.actions_units = [
            # Ordered by priority from highest to lowest
            PillageAction,
            SpawnCityAction,
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            # partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART), # Transfer to nearby cart
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER), # Transfer to nearby worker
        ]
        self.actions_cities = [
            # Ordered by priority from highest to lowest
            SpawnWorkerAction,
            ResearchAction,
            # SpawnCartAction,
        ]
        self.researched = {'wood': True, 'coal': False, 'uranium': False}
        self.resource_tiles = []

    def get_resource_tiles(self, game_map):
        self.resource_tiles = []
        for y in range(game_map.height):
            for x in range(game_map.width):
                cell = game_map.get_cell(x, y)
                if not cell.has_resource(): continue
                if cell.resource.type == 'coal' and not self.researched['coal']: continue
                if cell.resource.type == 'uranium' and not self.researched['uranium']: continue
                self.resource_tiles.append(cell)
        return self.resource_tiles

    def get_closest_resource(self, unit):
        closest_dist = float('inf')
        closest_resource_tile = None
        for resource_tile in self.resource_tiles:
            dist = resource_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_resource_tile = resource_tile
        return closest_resource_tile

    def get_direction_to_closest_resource(self, unit):
        closest_resource = self.get_closest_resource(unit)
        if not closest_resource:
            return None
        return unit.pos.direction_to(closest_resource.pos)

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        self.researched = game.state["teamStates"][team]['researched']
        self.get_resource_tiles(game.map)

        # shuffle move actions
        sublist = self.actions_units[2:]
        random.shuffle(sublist)
        self.actions_units[2:] = sublist

        actions = []
        actions_validated = []

        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            unit_actions = []

            if unit.get_cargo_space_left() > 0 and random.random() < 0.5: # decide whether to move to resource
                direction = self.get_direction_to_closest_resource(unit)
                if direction and direction != 'c':
                    action = self.move_action_makers[direction](game=game,
                                                                unit_id=unit.id,
                                                                unit=unit,
                                                                city_id=None,
                                                                citytile=None,
                                                                team=team,
                                                                x=unit.pos.x,
                                                                y=unit.pos.y)
                    if action.is_valid(game, actions_validated):
                        actions_validated.append(action)
                        actions.append(action)
                        continue

            for index, action_maker in enumerate(self.actions_units):
                action = action_maker(game=game,
                                      unit_id=unit.id,
                                      unit=unit,
                                      city_id=None,
                                      citytile=None,
                                      team=team,
                                      x=unit.pos.x,
                                      y=unit.pos.y)
                if action.is_valid(game, actions_validated):
                    actions_validated.append(action)
                    if type(action) is not PillageAction or random.random() < 0.5: # pillage at x% chance
                        actions.append(action)
                        break

        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    city_actions = []
                    for index, action_maker in enumerate(self.actions_cities):
                        action = action_maker(game=game,
                                              unit_id=None,
                                              unit=None,
                                              city_id=city_tile.city_id,
                                              citytile=city_tile,
                                              team=team,
                                              x=city_tile.pos.x,
                                              y=city_tile.pos.y)
                        if action.is_valid(game, actions_validated):
                            actions_validated.append(action)
                            if type(action) != ResearchAction or not self.researched['uranium']: # only research if not yet 200
                                actions.append(action)
                                break

        # print(actions)
        # print(len(units))
        return actions
