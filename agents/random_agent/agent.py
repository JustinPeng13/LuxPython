import sys
import time
from functools import partial  # pip install functools
import copy
import random
import numpy as np

from luxai2021.env.agent import Agent
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position

class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn.
        :param game:
        :param team:
        :return: Array of actions to perform for this turn.
        """
        actions = []
        return actions

    def pre_turn(self, game, is_first_turn=False):
        """
        Called before a turn starts. Allows for modifying the game environment.
        Generally only used in kaggle submission opponents.
        :param game:
        """
        return

    def post_turn(self, game, actions):
        """
        Called after a turn. Generally only used in kaggle submission opponents.
        :param game:
        :param actions:
        :return: (bool) True if it handled the turn (don't run our game engine)
        """
        return False

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
