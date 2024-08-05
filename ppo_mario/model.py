import torch
from stable_baselines3 import PPO
from .config import TrainConfiguration
from gymnasium import Env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import CnnPolicy


def set_freeze(model: torch.nn.Module, freeze: bool):
    for param in model.parameters():
        param.requires_grad = not freeze


def create_model(cfg: TrainConfiguration, env: Env | VecEnv = None) -> PPO:
    """Load the model from the given path."""

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device:", device)

    if cfg.base_model is None:
        model = PPO(
            CnnPolicy,
            env=env,
            device=device,
            **cfg.ppo_cfg,
        )
        print("[Model] Created a new model.")
    else:
        model = PPO.load(cfg.base_model, env=env, device=device, **cfg.ppo_cfg)
        print("[Model] Loaded from", cfg.base_model)

    set_freeze(model.policy.pi_features_extractor, cfg.freeze_actor)
    set_freeze(model.policy.vf_features_extractor, cfg.freeze_actor)
    set_freeze(model.policy.mlp_extractor.policy_net, cfg.freeze_actor)
    set_freeze(model.policy.action_net, cfg.freeze_actor)
    print("Actor frozen:", cfg.freeze_actor)
    print("Check:", next(model.policy.features_extractor.parameters()).requires_grad)

    return model
