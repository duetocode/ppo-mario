from typing import Tuple
from gymnasium import Env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO, CnnPolicy
from ppo_mario import get_device, TrainConfiguration
from ppo_mario.networks import ResNetFeatureExtractor


def create_model(
    env: Env | VecEnv | None = None, device: str | None = None
) -> Tuple[PPO, TrainConfiguration]:
    cfg = TrainConfiguration(
        batch_size=128,
        n_steps=2048,
        gamma=0.9,
        learning_rate=1e-4,
        target_kl=0.2,
        clip_range=0.1,
        normalize_advantage=False,
        freeze_actor=False,
    )

    model = PPO(
        CnnPolicy,
        env=env,
        **cfg.ppo_cfg,
        policy_kwargs={
            "features_extractor_class": ResNetFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "share_features_extractor": False,
            "net_arch": {"pi": [], "vf": []},
        },
        device=device if device is not None else get_device,
    )
    return model, cfg
