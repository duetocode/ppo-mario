import os, functools, sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import CheckpointCallback

from .misc import WorkDir, get_sub_process_start_method

from .config import TrainConfiguration
from .environment import create_env
from .model import create_model


def train(work_dir: WorkDir, n_envs: int = None):
    """Train the model within the given work directory"""

    # check the work directory first
    if not (
        work_dir is not None and work_dir.root.exists() and work_dir.config.exists()
    ):
        print(
            "Invalid work directory. Do you properly initialized it?", file=sys.stderr
        )
        sys.exit(-10)

    print("Will work in", str(work_dir.root))

    # normalize the n_envs
    n_envs = n_envs if n_envs else os.cpu_count()

    # load the configuration
    cfg = TrainConfiguration.load(work_dir.config.read_text())
    print(f"Will train {cfg.total_timesteps} steps with {n_envs} processes.")
    print("Configuration:")
    print(cfg.to_json())

    # prepare the environment
    venv = SubprocVecEnv(
        [
            functools.partial(
                create_env,
                with_random_frame_skip=cfg.random_frame_skip,
                level=tuple(cfg.level),
            )
        ]
        * n_envs,
        start_method=get_sub_process_start_method(),
    )
    venv = VecMonitor(venv)
    print("Observation space:", venv.observation_space)
    print("Action space:", venv.action_space)

    # prepare the model
    model = create_model(cfg, base_model=work_dir.base_model, env=venv)

    # setup tensorboard logging
    model.set_logger(logger.configure(str(work_dir.logs), ["tensorboard"]))

    # setup the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(work_dir.checkpoints),
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # train the model
    print("Training...")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        tb_log_name="training",
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(str(work_dir.saved_model))
    print("Model saved to", str(work_dir.saved_model))
    print("Done")
