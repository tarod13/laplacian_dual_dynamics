import numpy as np
from . import maze
from . import maze2d_single_goal
from .. import env_base


ONE_ROOM_MAZE = maze.Maze(maze.SquareRoomFactory(size=15))
# ONE_ROOM_GOAL_POS = np.array([15, 15])
TWO_ROOM_MAZE = maze.Maze(maze.TwoRoomsFactory(size=15))
# TWO_ROOM_GOAL_POS = np.array([9, 15])
FOUR_ROOM_MAZE = maze.Maze(maze.FourRoomsFactory(size=7))
# TWO_ROOM_GOAL_POS = np.array([9, 15])
ROOM64_MAZE = maze.Maze(maze.MazeFactoryBase(maze_str=maze.ROOM64_MAZE))
HARD_MAZE = maze.Maze(maze.MazeFactoryBase(maze_str=maze.HARD_MAZE))
# HARD_MAZE_GOAL_POS = np.array([10, 10])
CUSTOM_MAZE = maze.Maze(maze.MazeFactoryBase(maze_str=maze.CUSTOM_MAZE))


class OneRoomEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=ONE_ROOM_MAZE,
                episode_len=50,
                start_pos='random',
                use_stay_action=False,
                reward_type='neg',
                # goal_pos=ONE_ROOM_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)


class TwoRoomEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=TWO_ROOM_MAZE,
                episode_len=50,
                start_pos='random',
                use_stay_action=False,
                reward_type='neg',
                # goal_pos=TWO_ROOM_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)


class FourRoomEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=FOUR_ROOM_MAZE,
                episode_len=50,
                start_pos='random',
                use_stay_action=False,
                reward_type='neg',
                # goal_pos=TWO_ROOM_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)

class Room64Env(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=ROOM64_MAZE,
                episode_len=50,
                start_pos='random',
                use_stay_action=False,
                reward_type='neg',
                # goal_pos=HARD_MAZE_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)

class HardMazeEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=HARD_MAZE,
                episode_len=50,
                start_pos='random',
                use_stay_action=False,
                reward_type='neg',
                # goal_pos=HARD_MAZE_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)

class CustomMazeEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=CUSTOM_MAZE,
                episode_len=100,
                start_pos='random',
                use_stay_action=False,
                reward_type='neg',
                # goal_pos=HARD_MAZE_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)



ENV_CLSS = {
    'OneRoom': OneRoomEnv,
    'TwoRoom': TwoRoomEnv,
    'FourRoom': FourRoomEnv,
    'CustomMaze': CustomMazeEnv,
    'HardMaze': HardMazeEnv,
    'Room64': Room64Env,
}


def make(env_id):
    return ENV_CLSS[env_id]()

