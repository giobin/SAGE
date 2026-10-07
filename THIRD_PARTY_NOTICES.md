# Third-party notices

This repository includes code and assets adapted from the projects below. Their licenses apply to the corresponding files.

| Component | Source | License | Files |
|---|---|---|---|
| EZPoints environment and card images | [RL4VLM gym-cards](https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards) | MIT, see `src/environments/gym_cards/LICENSE.txt` | `src/environments/gym_cards/` (CardMaze is new in this work) |
| DejaVu Sans font | [DejaVu fonts](https://dejavu-fonts.github.io) | Bitstream Vera license, see `src/environments/gym_cards/envs/font/LICENSE-DejaVu` | `src/environments/gym_cards/envs/font/DejaVuSans.ttf` |
| PPO training loop | [CleanRL](https://github.com/vwxyzjn/cleanrl) | MIT | `src/train/train_ppo.py`, `src/agents/ac_mlp.py` |
| IMPALA CNN | [AIcrowd NeurIPS 2020 Procgen starter kit](https://github.com/AIcrowd/neurips2020-procgen-starter-kit) | Apache-2.0 | `src/agents/ac_cnn.py`, `src/agents/common.py` |
| Text and memory agent | [rl-starter-files](https://github.com/lcswillems/rl-starter-files), [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) | MIT | `src/agents/ac_cnn_text.py` |
| FrozenLake transition code | [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | MIT | `src/environments/wrappers.py` |
| ALFWorld configuration | [ALFWorld](https://github.com/alfworld/alfworld) | MIT | `config/env/alfworld.yaml` |

The LVLM2P and RL-VLM-F baselines are our own re-implementations from the papers. Gymnasium, MiniGrid, ALFWorld, vLLM, Transformers and the Qwen and Gemma model weights are installed or downloaded separately under their own licenses and terms.
