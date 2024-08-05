from pathlib import Path
import argparse


def run(model_file: Path):
    # delay the import to speedup the startup
    import ppo_mario

    # fix
    if model_file.is_dir():
        model_file = model_file / "model-supervised.zip"

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
    model = ppo_mario.create_model(cfg)

    # the output file should place in the same directory as the model
    output_file = model_file.parent / "gameplay.mp4"

    # render
    ppo_mario.render(model, output_file)

    # output
    print("The gameplay video is saved to", str(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a gameplay episode with the given model."
    )
    parser.add_argument("model", type=Path, help="The path to the model file.")

    args = parser.parse_args()

    print("Render a gameplay episode with the given model: ", str(args.model))

    run(args.model)
