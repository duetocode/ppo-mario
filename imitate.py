from pathlib import Path


def main(expert_data_dir: Path, epoches: int, learning_rate: float, batch_size: int):
    """Train the agent model with Imitation Learning"""
    import json, sys
    from ppo_mario.imitation import BehaviorCloning

    bc = BehaviorCloning(
        expert_data_dir=expert_data_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    bc.train(epoches)

    # ensure the model is saved
    if bc.saved_model.exists():
        print("Best model:")
        print(json.dumps(bc.best_model_info, indent=4))
        print("The model is saved to", str(bc.saved_model))
    else:
        print("I thought the model was saved, but it's not there.", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train an agent model with Imitation Learning.")
    parser.add_argument(
        "--expert_data_dir",
        "-d",
        type=Path,
        default="assets/playthrough/samples",
        help="The directory of the expert data.",
    )
    parser.add_argument(
        "--epoches", "-t", type=int, default=50, help="The number of epoches to train."
    )
    parser.add_argument(
        "--learning_rate", "-l", type=float, default=1e-3, help="The learning rate."
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="The batch size."
    )

    args = parser.parse_args()
    main(**vars(args))
