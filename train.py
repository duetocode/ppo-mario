import sys
from pathlib import Path
import shutil

from ppo_mario.misc import WorkDir

DEFAULT_CFG_FILE = "config.json"


def create_work_directory(work_dir: WorkDir, args):
    if work_dir.root.exists():
        print("The work directory already exists.", file=sys.stderr)
        sys.exit(-1)

    print("Creating a new working directory at ", work_dir)
    work_dir.mkdirs()
    print("Write default configuration to", DEFAULT_CFG_FILE)

    from ppo_mario import TrainConfiguration
    from ppo_mario.misc import copy_preivous_logs

    # the default configuration
    cfg = TrainConfiguration()
    # copy the base model if it is provided
    if args.base_model:
        base_model = Path(args.base_model)
        if not (base_model.exists() or base_model.is_file()):
            print("Invalid base model path.", file=sys.stderr)
            sys.exit(-2)
        # copy it here
        shutil.copy(base_model, work_dir.base_model)
        # copy the previous logs if they exist
        copy_preivous_logs(base_model, work_dir)

    # write the configuration to the file
    work_dir.config.write_text(cfg.to_json())

    print("Done.")


def main(work_dir: WorkDir, args):
    from ppo_mario import train
    from ppo_mario.misc import launch_tensorboard

    # check to avoid overwriting the work directory
    if not (work_dir.root.exists() and work_dir.root.is_dir()):
        print("Invalid work directory.", (work_dir.root), file=sys.stderr)
        sys.exit(-10)

    # check if the config file exists
    if not work_dir.config.exists():
        print("Configuration file not found.", file=sys.stderr)
        sys.exit(-20)

    # launch the tensorboard
    launch_tensorboard(work_dir.logs)

    # call the training function
    train(work_dir, n_envs=args.jobs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("work_dir", type=str, help="The work directory.")
    parser.add_argument(
        "--create", "-c", action="store_true", help="Create a new work directory."
    )
    parser.add_argument(
        "--base-model",
        "-m",
        type=Path,
        default=None,
        help="The base model for continuing training.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="The number of parallel processes for the rollouts.",
    )
    args = parser.parse_args()

    work_dir = WorkDir(args.work_dir)

    if args.create:
        create_work_directory(work_dir, args)
    else:
        main(work_dir, args)
