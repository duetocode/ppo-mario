from pathlib import Path

import torch
from stable_baselines3 import PPO

from ppo_mario.networks.feature_extractor import ResNetFeatureExtractor

from .config import TrainConfiguration
from gymnasium import Env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import CnnPolicy


def set_freeze(model: torch.nn.Module, freeze: bool):
    """Set the requires_grad attribute of the model."""
    for param in model.parameters():
        param.requires_grad = not freeze


def generate_model_cfg(cfg: TrainConfiguration) -> dict:
    """Create the parameters for the PPO constructor call."""
    ppo_cfg = cfg.ppo_cfg

    # convert the `policy_kwargs.feature_extractor_class` to the actual class
    policy_kwargs = ppo_cfg.setdefault("policy_kwargs", {})
    clazz = policy_kwargs.get("features_extractor_class", None)
    if clazz is None:
        # use the default, which is fine
        pass
    elif clazz == "ResNetFeatureExtractor":
        # use the custom feature extractor
        policy_kwargs["features_extractor_class"] = ResNetFeatureExtractor
    else:
        # unknown class
        raise ValueError(
            f"Unknown feature extractor class: {cfg.features_extractor_class}"
        )

    return ppo_cfg


def create_model(
    cfg: TrainConfiguration, base_model: Path | None = None, env: Env | VecEnv = None
) -> PPO:
    """Load the model from the given path."""

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device:", device)

    if base_model and base_model.exists():
        # load the model
        # remove the policy_kwargs, as it is not needed
        if "policy_kwargs" in cfg.ppo_cfg:
            del cfg.ppo_cfg["policy_kwargs"]
        # load the model
        model = PPO.load(str(base_model), env=env, device=device)
        print("[Model] Loaded from", str(base_model))
    else:
        # create a new model, with the given configuration
        # but first, we need to takecare the policy_kwargs parameter
        ppo_cfg = generate_model_cfg(cfg)
        model = PPO(
            CnnPolicy,
            env=env,
            device=device,
            **ppo_cfg,
        )
        print("[Model] Created a new model.")

    set_freeze(model.policy.pi_features_extractor, cfg.freeze_actor)
    set_freeze(model.policy.vf_features_extractor, cfg.freeze_actor)
    set_freeze(model.policy.mlp_extractor.policy_net, cfg.freeze_actor)
    set_freeze(model.policy.action_net, cfg.freeze_actor)
    print("Actor frozen:", cfg.freeze_actor)
    print(
        "Check requires_grad:",
        next(model.policy.features_extractor.parameters()).requires_grad,
    )

    return model
