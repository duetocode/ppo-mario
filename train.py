from pathlib import Path


def main(args):
    from ppo_mario import train, TrainConfiguration

    train(
        TrainConfiguration(
            model=str(args.model),
            total_timesteps=args.total_steps,
            learning_rate=args.learning_rate,
            freeze_actor=args.freeze_actor,
            n_steps=args.steps,
            random_frame_skip=args.random_frame_skip,
        ),
        n_envs=args.jobs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("model", type=Path, help="The path to the model file.")
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
