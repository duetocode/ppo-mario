from pathlib import Path
from datetime import datetime
import shutil


def main(args):
    from ppo_mario import train, TrainConfiguration
    from ppo_mario.misc import copy_preivous_logs, launch_tensorboard

    # prepare the work dir
    ts = datetime.now()
    work_dir = Path("work", ts.strftime("%Y-%m-%d_%H-%M"))
    if work_dir.exists():
        raise FileExistsError(f"Directory {work_dir} already exists")
    print("Work directory:", work_dir)

    # the configuration
    cfg = TrainConfiguration(
        work_dir=str(work_dir),
        base_model=args.base_model,
        total_timesteps=args.total_steps,
        learning_rate=args.learning_rate,
        freeze_actor=args.freeze_actor,
        n_steps=args.steps,
        random_frame_skip=args.random_frame_skip,
    )
    cfg.prepare_work_dir()

    # copy the previous logs if they exist
    if args.base_model:
        copy_preivous_logs(args.base_model, work_dir)

    # launch the tensorboard
    launch_tensorboard(cfg.logs_dir)

    # call the training function
    train(cfg, n_envs=args.jobs)

    # archive the work directory
    archive_dir = Path("archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(cfg.work_dir), str(archive_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--base-model", "-m", type=str, help="The path to the model file."
    )
    parser.add_argument("--jobs", "-j", type=int, default=8, help="The number of jobs.")
    parser.add_argument(
        "--total-steps",
        "-t",
        type=int,
        default=100_000,
        help="The total number of timesteps to train.",
    )
    parser.add_argument(
        "--learning-rate", "-l", type=float, default=2e-5, help="The learning rate."
    )
    parser.add_argument(
        "--freeze-actor",
        "-f",
        action="store_true",
        help="Freeze the actor network.",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=2048,
        help="The number of steps per simulation",
    )
    parser.add_argument(
        "--random-frame-skip",
        action="store_true",
        default=False,
        help="Randomize the frame skip",
    )
    args = parser.parse_args()

    main(args)
