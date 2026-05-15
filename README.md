# LLaMA-PPO-RND for Gym-DSSAT: Language Model-Enhanced Reinforcement Learning for Crop Management

<p align="center">
  <strong>Large Language Models meet Agricultural Reinforcement Learning</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-main-experiment">Main Experiment</a> •
  <a href="#-ablation-studies">Ablation Studies</a> •
  <a href="#-environment-setup">Environment Setup</a>
</p>

---

## Overview

This project explores using **Large Language Models (LLMs)** as state encoders within reinforcement learning agents for **optimal crop management decisions**. Built on the [Gym-DSSAT](https://github.com/GijsvanDijck/gym-dssat-pdi) crop simulation environment (based on the DSSAT crop growth model), our approach combines **PPO + Chinese-LLaMA-2 + RND (Random Network Distillation)** with **progressive augmentation** to achieve superior performance in irrigation and fertilization decisions.

Our main contribution — **PPO + LLaMA + RND with Progressive Augmentation** — significantly outperforms the SOTA baseline (DQN+BERT from [*The New Agronomists: Language Models are Experts in Crop Management*](https://arxiv.org/abs/2403.19839)), while comprehensive ablation studies validate the effectiveness of each component.

### Key Innovations

1. **LLaMA as State Encoder**: Leverages Chinese-LLaMA-2-1.3B to encode agricultural state observations into rich semantic embeddings, enabling the RL agent to reason about crop states in natural language space.

2. **RND Intrinsic Reward with Progressive Decay**: Employs Random Network Distillation to provide intrinsic exploration bonuses, with a gradual decay schedule that transitions the agent from exploration to exploitation.

3. **Progressive Augmentation Strategy**: Inspired by the SOTA paper's finding (Table 6: temperature noise ±2°C causes -11.9% performance drop), we introduce a three-phase augmentation:
   - **Phase 1 (0–40%)**: No augmentation — learn basic policy
   - **Phase 2 (40–60%)**: Gradually increase measurement noise
   - **Phase 3 (60–100%)**: Full augmentation — maximize generalization

4. **Policy Ensemble & Early Stopping**: Maintains a top-K checkpoint ensemble and uses multi-seed validation for robust model selection.

## Architecture

```
                          ┌──────────────────────┐
                          │  Gym-DSSAT (DSSAT)   │
                          │  Crop Simulator      │
                          └──────────┬───────────┘
                                     │ State (25-dim)
                                     ▼
                          ┌──────────────────────┐
                          │  State → Text        │
                          │  array2str()         │
                          └──────────┬───────────┘
                                     │ Text Description
                                     ▼
                    ┌────────────────────────────────┐
                    │  Chinese-LLaMA-2-1.3B Encoder  │
                    │  (BF16, Gradient Checkpointing)│
                    └────────────────┬───────────────┘
                                     │ Hidden States
                                     ▼
                    ┌────────────────────────────────┐
                    │  Projection Layer              │
                    │  (LLaMA hidden → 256-dim)      │
                    └────────────────┬───────────────┘
                                     │ Embedding (256-dim)
                          ┌──────────┴───────────┐
                          ▼                      ▼
               ┌─────────────────┐    ┌─────────────────┐
               │  Actor Head     │    │  Critic Head    │
               │  (Action Logits)│    │  (State Value)  │
               └────────┬────────┘    └─────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Action (25-dim)│
               │  anfer + amir   │
               └─────────────────┘

     ┌─────────────────────────────────────────────┐
     │  RND Module (Parallel)                      │
     │  Target Network (fixed) + Predictor Network │
     │  → Intrinsic Reward (with progressive decay)│
     └─────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────┐
     │  Progressive Augmentor (Training only)      │
     │  Temperature Noise │ Rainfall Noise │       │
     │  Soil Moisture Noise │ State Dropout        │
     └─────────────────────────────────────────────┘
```

## Project Structure

```
llama_ppo_rnd_in_gymDssat/
├── main/                           # All experiment implementations
│   ├── llama_ppo_rnd/              # ★ Main experiment
│   │   ├── llama_ppo_rnd_update_glm5_v3.py   # ★ Main: PPO+LLaMA+RND+Progressive Aug
│   │   ├── llama_ppo_rnd_update_glm5_v2.py   # v2: Progressive augmentation (pre-refinement)
│   │   ├── llama_ppo_rnd_update_glm5_v1.py   # v1: Deep optimization fix
│   │   └── llama_ppo_rnd.py                   # v0: Base PPO+LLaMA+RND
│   │
│   ├── bert_dqn/                   # ★ SOTA reproduction (DQN+BERT)
│   │   ├── bert_dqn.py             # DQN+DistilBERT (original reproduction)
│   │   └── bert_dqn_update.py      # DQN+BERT with measurement noise augmentation
│   │
│   ├── llama_dqn/                  # Ablation: DQN+LLaMA
│   │   └── llama_dqn.py
│   │
│   ├── llama_ppo/                  # Ablation: PPO+LLaMA (no RND)
│   │   └── llama_ppo.py
│   │
│   ├── llama_ppo_mcts/             # Ablation: PPO+LLaMA+MCTS+World Model
│   │   ├── llama_ppo_mcts.py       # Agri-Reasoner V2
│   │   └── llama_ppo_mcts_update.py # V2.1 with RND checkpoint adaptation
│   │
│   ├── only_dqn/                   # Ablation: Pure DQN (no LLM)
│   │   └── only_dqn.py
│   │
│   ├── only_ppo/                   # Ablation: Pure PPO (no LLM)
│   │   └── only_ppo.py
│   │
│   ├── only_ppo_with_rnd/          # Ablation: Pure PPO+RND (no LLM)
│   │   └── only_ppo_with_rnd.py
│   │
│   ├── only_dreamV3/               # Ablation: DreamerV3 (World Model)
│   │   └── only_deramV3.py
│   │
│   └── only_mbpo/                  # Ablation: MBPO (Model-Based Policy Opt)
│       └── only_mbpo.py
│
├── models/                         # Pre-trained language models
│   └── chinese-llama-2-1.3b/      # Chinese-LLaMA-2 1.3B model weights
│
├── checkpoints/                    # Training checkpoints for all experiments
├── results/                        # Training results (plots, Excel, PDF)
├── test/                           # Testing & validation scripts
│   ├── val_proj/                   # Model validation scripts
│   ├── output_standard/            # Standard output plotting utilities
│   └── LM_AG-main/                # Original SOTA paper code reference
│
└── data/logs/                      # DSSAT simulation logs
```

## Main Experiment

**File**: [`main/llama_ppo_rnd/llama_ppo_rnd_update_glm5_v3.py`](main/llama_ppo_rnd/llama_ppo_rnd_update_glm5_v3.py)

This is the core implementation of our proposed method. Key technical details:

### State Encoding
- 25-dimensional agricultural state → text description via `array2str()`
- Text tokenized (max_length=128) and encoded by Chinese-LLaMA-2-1.3B
- Mean pooling + Projection Layer → 256-dimensional embedding

### Action Space
- Discrete action space of size 25 (5 fertilizer levels × 5 irrigation levels)
- `anfer`: nitrogen fertilizer (0, 40, 80, 120, 160 kg/ha)
- `amir`: irrigation (0, 6, 12, 18, 24 mm)
- Domain constraints: no fertilization after day 250 (`state[0] >= 10000`), no irrigation above field capacity (`state[21] >= 1600`)

### Reward Function
Following the SOTA paper's definition:
```
r = k1 * yield - k2 * n_action - k3 * w_action    (at episode end)
r = -k2 * n_action - k3 * w_action                  (during episode)
```
where `k1=0.158, k2=0.79, k3=1.1`

### RND Intrinsic Reward
- **Target network**: fixed random network
- **Predictor network**: trained to match target outputs
- **Intrinsic reward**: MSE prediction error, normalized by running mean/std
- **Progressive decay**: fully active before 30% training progress, linearly decays between 30%–70%, disabled after 70%

### Progressive Augmentation
Based on the SOTA paper's Table 6 analysis:

| Augmentation Type | Impact | Activation Phase |
|---|---|---|
| Temperature Noise (±2°C) | -11.9% (most impactful) | >30% progress |
| Rainfall Noise (5% std) | -3.2% (secondary) | >50% progress |
| Soil Moisture Noise (±0.02) | Minimal | >70% progress |
| State Dropout (5%, 1 feature) | Minimal | Probabilistic |

### Hyperparameters

| Parameter | Value |
|---|---|
| Training Episodes | 2000 |
| Max Steps per Episode | 200 |
| Gamma (discount) | 0.99 |
| GAE Lambda | 0.95 |
| PPO Clip Ratio | 0.2 |
| PPO Epochs | 5 |
| Mini-batch Size | 64 |
| Update Frequency | 10 episodes |
| Actor LR | 5e-5 |
| Critic LR | 1e-4 |
| LLaMA LR | 1e-5 |
| Projection LR | 1e-4 |
| Intrinsic Coef | 0.01 |
| Embedding Dim | 256 |
| Precision | BF16 |

### Run

```bash
# Set your model path and output directory in the script, then:
cd main/llama_ppo_rnd/
python llama_ppo_rnd_update_glm5_v3.py
```

## Ablation Studies

All ablation experiments share the same Gym-DSSAT environment and reward function, isolating individual components to evaluate their contribution.

| Experiment | File | State Encoder | RL Algorithm | Exploration | Special |
|---|---|---|---|---|---|
| **Main (Ours)** | `llama_ppo_rnd/llama_ppo_rnd_update_glm5_v3.py` | LLaMA-1.3B | PPO | RND | Progressive Aug |
| **SOTA Baseline** | `bert_dqn/bert_dqn.py` | DistilBERT | DQN | ε-greedy | Paper reproduction |
| **DQN+LLaMA** | `llama_dqn/llama_dqn.py` | LLaMA-1.3B | DQN | ε-greedy | Encoder comparison |
| **PPO+LLaMA** | `llama_ppo/llama_ppo.py` | LLaMA-1.3B | PPO | — | RND ablation |
| **PPO+LLaMA+MCTS** | `llama_ppo_mcts/llama_ppo_mcts.py` | LLaMA+LoRA | PPO+MCTS | World Model | Agri-Reasoner V2 |
| **Pure DQN** | `only_dqn/only_dqn.py` | None (numeric) | DQN | ε-greedy | No LLM |
| **Pure PPO** | `only_ppo/only_ppo.py` | None (numeric) | PPO | — | No LLM |
| **PPO+RND** | `only_ppo_with_rnd/only_ppo_with_rnd.py` | None (numeric) | PPO | RND | No LLM, RND only |
| **DreamerV3** | `only_dreamV3/only_deramV3.py` | RSSM | Actor-Critic | World Model | Model-based RL |
| **MBPO** | `only_mbpo/only_mbpo.py` | Ensemble | SAC | Model Rollouts | Model-based RL |

### Ablation Insights

1. **LLM vs No-LLM**: LLaMA-1.3B encoding consistently outperforms raw numerical input, confirming language models capture richer state representations for agricultural decision-making.

2. **PPO vs DQN**: PPO provides more stable policy optimization with LLM encoders, while DQN's experience replay can be incompatible with fine-tuning LLM embeddings.

3. **RND Contribution**: RND intrinsic rewards significantly improve exploration in sparse-reward crop management scenarios, especially in early training phases.

4. **Progressive Augmentation**: Full augmentation from the start hurts convergence; our progressive strategy achieves both strong training performance and generalization.

5. **Model-Based Methods**: DreamerV3 and MBPO underperform in this domain, likely due to the difficulty of accurately modeling DSSAT's complex biophysical processes.

## SOTA Baseline Reproduction

**File**: [`main/bert_dqn/bert_dqn.py`](main/bert_dqn/bert_dqn.py)

Reproduction of the method from:
> *The New Agronomists: Language Models are Experts in Crop Management* (arXiv:2403.19839)
> Original code: [jingwu6/LM_AG](https://github.com/jingwu6/LM_AG)

Key implementation details matched with the original paper:
- DistilBERT as state encoder (`distilbert-base-uncased`)
- DQN with experience replay (buffer size 1e5, batch size 512)
- Hard target network update (τ=8)
- ε-greedy exploration (decay=0.994, end=0.01)
- Gradient clipping (-1, 1)
- Terminal state experience duplication (7×)
- Learning rate: 1e-5

The updated version (`bert_dqn_update.py`) adds measurement noise augmentation as described in the paper's Table 6.

## Metrics

All experiments track both agricultural and AI metrics:

### Agricultural Metrics
| Metric | Description | Unit |
|---|---|---|
| Final Yield | Crop yield at harvest | kg/ha |
| Irrigation | Total water applied | mm |
| Fertilizer | Total nitrogen applied | kg/ha |
| WUE | Water Use Efficiency (Yield/Irrigation) | kg/mm |
| NUE | Nitrogen Use Efficiency (Yield/Fertilizer) | kg/kg |

### AI Metrics
| Metric | Description |
|---|---|
| Average Return | Mean cumulative reward |
| Sample Efficiency | Steps to reach expert-level performance |
| Convergence Speed | Steps until reward variance stabilizes |
| Exploration Coverage | Percentage of state space explored (via RND) |
| Generalization Gap | Train reward - Validation reward |

### Output Formats
Results are automatically saved as:
- **Excel** (`.xlsx`): Per-episode metrics + summary statistics
- **PDF** (`.pdf`): Formatted report with tables
- **PNG** (`.png`): Training curves and metric plots

## Environment Setup

### Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with ≥16GB VRAM (A100 40GB recommended for LLaMA fine-tuning)
- **Python**: 3.8+
- **DSSAT**: DSSAT-PDI installed at `/opt/dssat_pdi/`

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install gym==0.21.0

# Data processing & visualization
pip install pandas openpyxl matplotlib reportlab

# Progress bars
pip install tqdm

# Gym-DSSAT (follow official installation guide)
# https://github.com/GijsvanDijck/gym-dssat-pdi
```

### Install Gym-DSSAT

```bash
# Follow the official gym-dssat-pdi installation guide
git clone https://github.com/GijsvanDijck/gym-dssat-pdi.git
cd gym-dssat-pdi
pip install -e .
```

### Download LLaMA Model

Download [Chinese-LLaMA-2-1.3B](https://huggingface.co/FlagAlpha/Chinese-LLaMA-2-1.3B) and place it under `models/chinese-llama-2-1.3b/`.

### Configuration

Before running, update the following paths in the script:

```python
# Model path
model_path = '/path/to/chinese-llama-2-1.3b'

# DSSAT binary
run_dssat_location = '/opt/dssat_pdi/run_dssat'

# Output directories
output_dir = '/path/to/results'
checkpoint_dir = '/path/to/checkpoints'
```

## Model Validation

Use the validation scripts to evaluate trained checkpoints:

```bash
# Validate a specific checkpoint
cd test/val_proj/
python valodate_llama_ppo_rnd.py \
    --checkpoint /path/to/checkpoints/best/model_ep1600.pth \
    --n_episodes 100

# Validate with stochastic policy
python valodate_llama_ppo_rnd.py \
    --stochastic \
    --n_episodes 100

# Validate BERT-DQN
python val_bert_dqn.py \
    --checkpoint /path/to/bert_dqn_checkpoints/model1520.pth

# Validate LLaMA-PPO-MCTS (from RND checkpoint)
python validate_llama_ppo_mcts_from_rnd.py \
    --checkpoint /path/to/llama_ppo_rnd_checkpoints/best
```

## Reward Function

The reward function follows the SOTA paper's definition, balancing yield maximization with resource conservation:

```python
def get_reward(state, n_action, w_action, next_state, done, k1=0.158, k2=0.79, k3=1.1):
    if done:
        return k1 * state[4] - k2 * n_action - k3 * w_action  # Terminal: yield - costs
    return -k2 * n_action - k3 * w_action                        # Step: only costs
```

- `state[4]`: final crop yield (kg/ha)
- `n_action`: nitrogen fertilizer amount (kg/ha)
- `w_action`: irrigation amount (mm)
- `k1=0.158, k2=0.79, k3=1.1`: reward coefficients from the paper

## Citation

If you find this work useful, please cite:

```bibtex
@article{thenewagronomists2024,
  title={The New Agronomists: Language Models are Experts in Crop Management},
  author={Jing Wu and others},
  journal={arXiv preprint arXiv:2403.19839},
  year={2024}
}
```

## References

1. **SOTA Paper**: Wu et al. "The New Agronomists: Language Models are Experts in Crop Management", arXiv:2403.19839
2. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms", 2017
3. **RND**: Burda et al. "Exploration by Random Network Distillation", ICLR 2019
4. **DreamerV3**: Hafner. "Mastering Diverse Domains through World Models", 2023
5. **MBPO**: Janner et al. "When to Trust Your Model: Model-Based Policy Optimization", NeurIPS 2019
6. **Chinese-LLaMA-2**: FlagAlpha, Chinese-LLaMA-2-1.3B
7. **Gym-DSSAT**: van Dijck et al., gym-dssat-pdi

## License

This project is released under the MIT License.

## Acknowledgements

- [Gym-DSSAT-PDI](https://github.com/GijsvanDijck/gym-dssat-pdi) for the crop simulation environment
- [LM_AG](https://github.com/jingwu6/LM_AG) for the SOTA baseline code
- [FlagAlpha](https://huggingface.co/FlagAlpha) for the Chinese-LLaMA-2 model
