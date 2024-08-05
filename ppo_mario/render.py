from pathlib import Path
import numpy as np, cv2
import torch

from gymnasium import Env
from stable_baselines3.ppo import PPO

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


def render_attention(image: np.ndarray, attention_map: torch.Tensor):
    # resize the attention map to match the image size
    # howerver, there should be 16 pixel padding on the top
    attention = cv2.resize(
        (attention_map * 255)[0].detach().cpu().numpy().astype(np.uint8).squeeze(),
        dsize=(256, 240 - TOP_PADDING),
        interpolation=cv2.INTER_CUBIC,
    )

    # apply color map
    attention = cv2.applyColorMap(attention, cv2.COLORMAP_JET)

    # prepare the canvas
    canvas = image.copy()
    # draw the attention map
    canvas[TOP_PADDING:, :] = canvas[TOP_PADDING:, :] * 0.5 + attention * 0.5

    return canvas


def render(model: PPO, output_file: str | Path, with_attention: bool = False):
    """Render a gameplay episode with the given model and output to the specified file."""
    # prepare the environment
    env = create_env(
        with_random_episode=False, with_frame_skip=False, with_mario_reward=False
    )
    obs, _ = env.reset()
    vf_features_extractor = model.policy.vf_features_extractor.train(False)

    # the video encoder
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        60.0,
        (WIDTH * 4, HEIGHT),
        True,
    )

    # render loop
    done = False
    frame = 0
    while not done:
        # preprocess the observation
        obs = np.transpose(np.squeeze(obs), (1, 2, 0))
        # make a prediction
        action, _ = model.predict(obs, deterministic=True)
        # run the value features extractor for its attention map
        with torch.no_grad():
            vf_features_extractor(
                torch.as_tensor(obs[None].transpose(0, 3, 1, 2)).to(model.device)
                / 255.0
            )

        # step the game
        for _ in range(3):
            obs, _, terminated, truncated, info = env.step(action.tolist())
            obs = np.asarray(obs)
            done = terminated or truncated
            if done:
                break
            # render the frame
            screen = env.render()
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            # render the attention views
            policy_attention = render_attention(
                screen, model.policy.features_extractor._attention
            )
            value_attention = render_attention(screen, vf_features_extractor._attention)

            # create the output screen
            canvas = np.zeros((HEIGHT, WIDTH * 4, 3), dtype=np.uint8)
            # draw the screen on the top
            canvas[:, WIDTH * 0 : WIDTH * 1, :] = screen
            canvas[:, WIDTH * 1 : WIDTH * 2, :] = policy_attention
            canvas[:, WIDTH * 2 : WIDTH * 3, :] = value_attention
            # draw the information
            draw_info(canvas, info, frame)
            # save the frame
            writer.write(canvas)

            frame += 1

    writer.release()
