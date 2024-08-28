from time import time
from pathlib import Path
import numpy as np, cv2
import torch

from gymnasium import Env
from stable_baselines3.ppo import PPO

from ppo_mario.config import TrainConfiguration

from .environment import create_env

WIDTH = 256
HEIGHT = 240
TOP_PADDING = 16 * 2


def put_text(img, text, position):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_info(canvas: np.ndarray, info: dict, frame: int):
    put_text(canvas, f"Frame: {frame}", (768 + 30, 20))
    put_text(canvas, f"x:{info['x_pos']}", (768 + 30, 40))
    put_text(canvas, f"y:{info['y_pos']}", (768 + 30, 60))


def render(
    model: PPO,
    output_file: str | Path,
    n_frame_skipping: int,
    cfg: TrainConfiguration,
    with_attention: bool = False,
) -> int:
    """
    Render a gameplay episode with the given model and output to the specified file.

    returns
    -------
    int
        The number of frames rendered.
    """
    # prepare the environment
    env = create_env(
        with_random_episode=False,
        with_frame_skip=False,
        with_mario_reward=False,
        level=tuple(cfg.level),
    )
    obs, _ = env.reset()
    features_extractor = model.policy.features_extractor

    # the video encoder
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        60.0,
        (WIDTH, HEIGHT),
        True,
    )

    # render loop
    inference_times = []
    done = False
    frame = 0
    while not done:
        # preprocess the observation
        obs = np.transpose(np.squeeze(obs), (1, 2, 0))
        # make a prediction
        t_0 = time()
        action, _ = model.predict(obs, deterministic=True)
        inference_times.append(time() - t_0)

        # get the attention map from the spatial gate attention layer of the features extractor
        attention_map = None
        if hasattr(features_extractor, "attention_data"):
            # get the attention map
            attention_map = features_extractor.attention_data
            # convert to in RAM image
            attention_map = (
                (attention_map * 255).cpu().numpy().astype(np.uint8).squeeze()
            )
            # resize the attention map to match the image size
            attention_map = cv2.resize(
                attention_map,
                dsize=(256, 240 - TOP_PADDING),
                interpolation=cv2.INTER_CUBIC,
            )
            # color map
            attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

        # step the game
        for _ in range(n_frame_skipping):
            obs, _, terminated, truncated, info = env.step(action.tolist())
            obs = np.asarray(obs)
            done = terminated or truncated
            if done:
                break
            # render the frame
            screen = env.render()
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            # render the attention views
            if attention_map is not None:
                # draw the attention map over the screen
                screen[TOP_PADDING:, :] = (
                    screen[TOP_PADDING:, :] * 0.5 + attention_map * 0.5
                )

            writer.write(screen)

            frame += 1

    print("Average inference time:", np.mean(inference_times[10:]))

    writer.release()
    return frame
