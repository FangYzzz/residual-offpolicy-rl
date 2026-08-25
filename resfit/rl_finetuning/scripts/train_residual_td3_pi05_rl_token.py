"""RL-Token residual TD3 entry point.

The shared trainer executes: collect frozen pi0 vision-language embeddings,
train/freeze the VAE bottleneck, construct token-based replay caches, then run
the unchanged residual TD3 collector/learner stages.
"""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from resfit.rl_finetuning.scripts.train_residual_td3_pi05_parallel_lang_chunk import main


@hydra.main(version_base=None, config_name="residual_td3_franka_complex_config")
def hydra_entry(cfg: ResidualTD3DexmgConfig):
    cfg.rl_token.enabled = True
    cfg.agent.language.enabled = False
    main(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    hydra_entry()
