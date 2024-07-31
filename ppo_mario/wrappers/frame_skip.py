import numpy as np
from gymnasium import Wrapper, Env


class FrameSkip(Wrapper):
    """A wrapper that skips a number of frames wth the specified action and returns the accumulated reward."""

    def __init__(self, env: Env, frame_skip: int = 4, jitter: bool = False):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.jitter = jitter
        self.rng = np.random.default_rng()

    def step(self, action: int) -> tuple:
        """Execute the action for a number of frames and return the accumulated reward."""
        # accumulated reward
        total_reward = 0
        # step the environment for the specified number of frames
        steps_to_skip = self.frame_skip
        if self.jitter:
            # add a random jitter to the frame skip
            # the noise is a normal distribution
            # and the result steps to skip is in [1, 8], closed interval
            steps_to_skip -= round(
                abs(self.rng.standard_normal() * (self.frame_skip / 2))
            )
            steps_to_skip = max(1, min(self.frame_skip, steps_to_skip))

        for _ in range(steps_to_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # accumulate the reward
            total_reward += reward
            # break if the game is ended
            if terminated or truncated:
                break

        # return the latest data with the accumulated reward
        return obs, total_reward, terminated, truncated, info
