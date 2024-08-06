from pathlib import Path


def main(
    expert_data_dir: str,
    work_dir: str,
    epoches: int,
    learning_rate: float,
    batch_size: int,
):
    """Train the agent model with Imitation Learning"""
    import json, sys
    from ppo_mario.imitation import BehaviorCloning
    from ppo_mario import TrainConfiguration
    from ppo_mario.misc import WorkDir

    work_dir = WorkDir(work_dir)
    # check if the work directory exists
    if not (work_dir.root.exists()):
        print("Work directory not found.", file=sys.stderr)
        sys.exit(-1)
    # and the expert data dir
    if not Path(expert_data_dir).exists():
        print("Expert data directory not found.", file=sys.stderr)
        sys.exit(-2)

    # run the training
    cfg = TrainConfiguration.load(work_dir.config.read_text())
    bc = BehaviorCloning(
        cfg=cfg,
        expert_data_dir=expert_data_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_save_path=work_dir.base_model,
    )
    bc.train(epoches)

    # ensure the model is saved
    if bc.model_save_path.exists():
        print("Best model:")
        print(json.dumps(bc.best_model_info, indent=4))
        print("The model is saved to", str(bc.model_save_path))
    else:
        print("I thought the model was saved, but it's not there.", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train an agent model with Imitation Learning.")
    parser.add_argument(
        "expert_data_dir", type=str, help="The directory of the expert data."
    )
    parser.add_argument("work_dir", type=str, help="The work directory.")
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
