from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
)
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from ppo_mario.wrappers import RandomEpisode, MarioReward, ObservationClip, FrameSkip
from ppo_mario.action_space import FAST_MOVE


def create_env(
    with_frame_skip: bool = True,
    with_random_frame_skip: bool = False,
    with_random_episode: bool = True,
    with_mario_reward: bool = True,
    level: tuple = (4, 1),
):
    # the basic environment creation
    env = gym_super_mario_bros.make(
        "SuperMarioBros-v0", target=level, render_mode="rgb_array"
    )

    # random episode reset
    if with_random_episode:
        env = RandomEpisode(env, data_dir="assets/playthrough")

    # new reward scheme and stuck detection
    if with_mario_reward:
        env = MarioReward(env, max_stuck_frames=16)

    # the action space for speedrunning
    env = JoypadSpace(env, actions=FAST_MOVE)

    # observation transformations

    # grayscale and resize for computational efficiency
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(120, 128))

    # clip to hide the score and time to prevent potential overfitting
    env = ObservationClip(env, top=16)

    # frame stacking for temporal information
    # use 8 because the NES is running at 60 FPS,
    # where the atari console is running at 30 FPS and the default is 4
    env = FrameStack(env, num_stack=8)

    # skip frames to reduce the computational cost
    if with_frame_skip:
        env = FrameSkip(env, frame_skip=8, jitter=with_random_frame_skip)

    return env
