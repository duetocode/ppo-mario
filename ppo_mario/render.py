from pathlib import Path
import numpy as np, cv2

from gymnasium import Env
from stable_baselines3.ppo import PPO

from .environment import create_env

WIDTH = 512
HEIGHT = 240


def put_text(img, text, position):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def render(model: PPO, output_file: str | Path, with_attention: bool = False):
    """Render a gameplay episode with the given model and output to the specified file."""
    # prepare the environment
    env = create_env(
        with_random_episode=False, with_frame_skip=False, with_mario_reward=False
    )
    obs, _ = env.reset()

    # the video encoder
    writer = cv2.VideoWriter(
        str(output_file), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (WIDTH, HEIGHT), True
    )

    # render loop
    done = False
    frame = 0
    while not done:
        # preprocess the observation
        obs = np.transpose(np.squeeze(obs), (1, 2, 0))
        # make a prediction
        action, _ = model.predict(obs, deterministic=True)
        # step
        for _ in range(8):
            obs, _, terminated, truncated, info = env.step(action.tolist())
            done = terminated or truncated
            if done:
                break
            # render the frame
            screen = env.render()
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            # create the output screen
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            # draw the screen on the top
            canvas[0:240, 0:256, :] = screen
            # draw the frame number
            put_text(canvas, f"Frame: {frame}", (270, 20))
            put_text(canvas, f"x:{info['x_pos']}", (270, 40))
            put_text(canvas, f"y:{info['y_pos']}", (270, 60))
            frame += 1

            writer.write(canvas)

    writer.release()
