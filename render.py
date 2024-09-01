from pathlib import Path
import argparse
from time import time


def run(model_file: Path, frame_skip: int, blur: bool = False):
    # delay the import to speedup the startup
    import ppo_mario
    import numpy as np

    # fix the path if it is needed
    if model_file.is_dir():
        model_file = model_file / "model.zip"

    # load the configuration
    cfg_file = model_file.parent / "config.json"
    if cfg_file.exists():
        cfg = ppo_mario.TrainConfiguration.load(
            (model_file.parent / "config.json").read_bytes()
        )
        print("Loading model with configuration:")
        print(cfg.to_json())
    else:
        # use default
        cfg = ppo_mario.TrainConfiguration()
        print("Loading model with DEFAULT configuration:")
    # but override the model file
    cfg.base_model = str(model_file)
    model = ppo_mario.create_model(cfg, base_model=model_file)

    # render
    for frame_skip in range(frame_skip, max(0, frame_skip - 1), -1):
        print(f"Rendering with frame skipping {frame_skip}...")
        t_0 = time()
        n_frames = ppo_mario.render(
            model,
            model_file.parent / f"gameplay_{frame_skip}.mp4",
            cfg=cfg,
            n_frame_skipping=frame_skip,
            blur=blur,
        )
        t_1 = time()
        print(
            f"Rendered {n_frames} for {n_frames / 60:.3f}s with frame skipping {frame_skip} in {t_1-t_0:.2f}s:"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a gameplay episode with the given model."
    )
    parser.add_argument("model", type=Path, help="The path to the model file.")
    parser.add_argument(
        "--frame-skip",
        "-s",
        type=int,
        default=-1,
        help="The frame skipping value. The default is to render all frame skipping values [1, 8].",
    )
    parser.add_argument(
        "--blur",
        "-b",
        action="store_true",
        default=False,
        help="Apply blur filter to the screenshots.",
    )

    args = parser.parse_args()

    print("Render a gameplay episode with the given model: ", str(args.model))

    run(args.model, args.frame_skip, args.blur)
