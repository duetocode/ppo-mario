import shutil, os, functools
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import CheckpointCallback

from .misc import copy_preivous_logs, launch_tensorboard, get_sub_process_start_method

from .config import TrainConfiguration
from .environment import create_env
from .model import create_model


def train(cfg: TrainConfiguration, n_envs: int = None):
    n_envs = n_envs if n_envs else os.cpu_count()
    print(f"Will train {cfg.total_timesteps} steps with {n_envs} processes.")
    print("Configuration:")
    print(cfg.save())

    # prepare the environment
    venv = SubprocVecEnv(
        [functools.partial(create_env, with_random_frame_skip=cfg.random_frame_skip)]
        * n_envs,
        start_method=get_sub_process_start_method(),
    )
    venv = VecMonitor(venv)
    print("Observation space:", venv.observation_space)
    print("Action space:", venv.action_space)

    # prepare the model
    model = create_model(cfg, env=venv)

    # setup tensorboard logging
    model.set_logger(logger.configure(str(cfg.logs_dir), ["tensorboard"]))

    # setup the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(cfg.checkpoints_dir),
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
    model.save(str(cfg.saved_model))
    print("Done")
