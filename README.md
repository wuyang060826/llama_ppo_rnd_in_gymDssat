# LLaMA-PPO-RND for Gym-DSSAT

An agricultural reinforcement learning research project for learning **irrigation and nitrogen fertilization policies** in the DSSAT crop simulator. The main experiment combines a LLaMA state encoder, PPO, Random Network Distillation (RND), and progressive observation augmentation.

[Results demo](#results-demo) · [Method](#method) · [Training](#training) · [Repository map](#repository-map)

## Results demo

Run this from the repository root with Python 3.9+. It summarizes the archived checkpoint evaluations with no installed packages, model download, GPU, or simulator.

```bash
python scripts/demo_results.py
```

![Archived batch checkpoint validation comparison](test/val_proj/batch_val_results_0417/validation_comparison_20260419_053122.png)

The figure and [raw CSV](test/val_proj/batch_val_results_0417/batch_validation_results_20260419_053122.csv) are archived experiment outputs. The demo averages each metric over the recorded repeats for each checkpoint. These data support inspection of checkpoint behavior in this batch. A matched evaluation across baseline methods and held-out seeds is needed to establish comparative performance.

## Method

```mermaid
flowchart LR
    DSSAT["DSSAT observation"] --> Text["State serialization"]
    Text --> LLaMA["LLaMA encoder"]
    LLaMA --> PPO["PPO actor–critic"]
    PPO --> Action["Irrigation and nitrogen"]
    Action --> DSSAT
    DSSAT --> RND["RND exploration bonus"]
    RND --> PPO
```

| Component | Implementation |
| --- | --- |
| State representation | Serialize the numerical observation as text and project LLaMA hidden states into the policy input. |
| Action space | 25 discrete combinations of five irrigation and five nitrogen application levels, subject to state-based application guards. |
| Extrinsic objective | Deduct irrigation and fertilizer costs at each step and credit the recorded yield at episode termination. |
| Intrinsic objective | Use RND prediction error to encourage exploration. |
| Progressive augmentation | Keep observations unchanged through 40% of training, ramp augmentation from 40% to 60%, then apply full strength. |

## Training

The main entry point uses the legacy Gym API and requires a Linux environment with an executable DSSAT-PDI runner, a BF16-capable NVIDIA GPU, and LLaMA weights compatible with Transformers. The archived experiment used a local Chinese-LLaMA-2-1.3B model directory.

1. Install a CUDA-enabled PyTorch build for your system using the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
2. Install the Python dependencies. Gym 0.21 may require legacy packaging tools in a dedicated Python 3.9 environment.

   ```bash
   python -m pip install "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
   python -m pip install --no-build-isolation -r requirements.txt
   ```

3. Install the [Gym-DSSAT-PDI fork and simulator](https://github.com/GijsvanDijck/gym-dssat-pdi) following its setup instructions.
4. Run a one-episode setup check with your local model and runner paths.

   ```bash
   python main/llama_ppo_rnd/llama_ppo_rnd_update_glm5_v3.py \
     --model /path/to/chinese-llama-2-1.3b \
     --dssat-runner /opt/dssat_pdi/run_dssat \
     --output-dir runs/smoke \
     --episodes 1
   ```

For the configured 2,000-episode experiment, omit `--episodes` and choose a fresh output directory. The run writes simulator logs, metric exports, and qualifying checkpoints under that directory. A one-episode run checks setup and runs final validation. PPO updates and checkpoint saves depend on thresholds in the training script. This environment setup has not been verified end to end.

## Repository map

| Path | Purpose |
| --- | --- |
| [main/llama_ppo_rnd](main/llama_ppo_rnd) | Main PPO + LLaMA + RND implementation and earlier variants. Use the v3 script for the command above. |
| [main/bert_dqn](main/bert_dqn) | BERT-based DQN reproduction experiments. |
| [main](main) | Additional PPO, DQN, RND, MCTS, DreamerV3, and MBPO experiment scripts. |
| [test/val_proj](test/val_proj) | Historical checkpoint evaluation scripts and outputs. Some scripts use experiment-specific local paths. |
| [test/LM_AG-main](test/LM_AG-main) | Upstream reference notebooks and assets from [LM_AG](https://github.com/jingwu6/LM_AG). |
| [scripts/demo_results.py](scripts/demo_results.py) | Dependency-free demo of the checked-in validation data. |

## References and reuse

- [The New Agronomists](https://arxiv.org/abs/2403.19839) and its [LM_AG repository](https://github.com/jingwu6/LM_AG) provide the language-model crop management baseline.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) and [Random Network Distillation](https://arxiv.org/abs/1810.12894) describe the learning methods used here.
- The repository currently has no standalone license file. Check the applicable upstream terms for third-party notebooks, model weights, and simulator components before redistribution.
