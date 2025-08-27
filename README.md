<h1 align="center">
<br>
rStar-Math
</h1>

<p align="center">
📃 <a href="https://huggingface.co/papers/2501.04519" target="_blank">[Paper]</a> 
</p>

Repo for "[rStar-Math: Small LLMs Can Master Math Reasoning
with Self-Evolved Deep Thinking](https://huggingface.co/papers/2501.04519)".

Authors: [Xinyu Guan](https://gxy-2001.github.io/)\*, [Li Lyna Zhang](https://www.microsoft.com/en-us/research/people/lzhani/)\*, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, Mao Yang

<p align="center">
    <img src="images/main_table.png" width="1000">
        <br>
    <em>Table 1: rStar-Math enables frontier math reasoning in SLMs via deep thinking over 64 trajectories.</em>
</p>

## News 

- **[07/15/2025]** Our rStar-Coder [paper](https://arxiv.org/abs/2505.21297) and [dataset](https://huggingface.co/datasets/microsoft/rStar-Coder) are released. We introduce a large-scale, verified dataset of 418K competition-level code problems with **test cases** of varying difficulty, enabling small LLMs (1.5B-14B) to achieve frontier-level code reasoning performance.
- **[02/10/2025]** We are hiring interns! If you are interested in improving LLM reasoning, please send your CV to lzhani@microsoft.com.
- **[01/21/2025]** Our code has been open-sourced. 
- **[01/09/2025]** Our paper is released: https://huggingface.co/papers/2501.04519.


Note: Our prior work [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://huggingface.co/papers/2408.06195) is open-sourced on the `rStar-mutualreasoning` branch.



## Contents
- [Introduction](#Introduction)
- [rStar2-Agent-RL-Training](#rStar2-Agent-RL-Training)
- [Citation](#Citation)


## Introduction
We present rStar-Math to demonstrate that small language models (SLMs) can rival or even surpass the math reasoning capability of OpenAI o1-mini, without distillation from superior models. rStar-Math achieves this by exercising "deep thinking" through Monte Carlo Tree Search (MCTS), where a math policy SLM performs test-time search guided by an SLM-based process reward model. The diagram below presents an overview of the rStar-Math framework, highlighting its core components and processes.

<p align="center">
  <img src="images/rstar.png">
</p>

## rStar2-Agent-RL-Training

A comprehensive reinforcement learning training framework for the rStar2-Agent, built on [Verl](https://github.com/volcengine/verl) and [Code Judge](https://github.com/0xWJ/code-judge). This framework enables training models after instruction-following supervised fine-tuning (SFT).

### Prerequisites

- Python 3.12
- Redis server
- GPU environment (8x A100/H100 recommended)
- Docker (recommended for isolated execution)

### Installation

#### Option 1: Manual Installation

```bash
# Initialize and update submodules
git submodule init
git submodule update

# Install Verl dependencies
pip install -r verl/requirements_sglang.txt
pip install -e verl

# Install Code Judge dependencies
pip install -r code-judge/requirements.txt
pip install -e code-judge

# Install rStar2-Agent requirements
pip install -r rstar2_agent/requirements.txt
```

#### Option 2: Automated Installation

```bash
bash install.sh
```

### Code Judge Server Setup

> ⚠️ **Security Warning**: Code Judge executes arbitrary code. Always deploy in an isolated environment (preferably Docker) and never expose to external networks.

The rStar2-Agent uses Code Judge as a tool call server to execute model-generated Python code.

#### 1. Start Redis Server

```bash
redis-server --daemonize yes --protected-mode no --bind 0.0.0.0
```

#### 2. Launch Code Judge Server

```bash
# Start the main server (master node only)
# Environment variables can be configured as per: https://github.com/0xWJ/code-judge/blob/main/app/config.py
# Replace $WORKSPACE and $MASTER_ADDR with your actual paths

tmux new-session -d -s server \
  'cd $WORKSPACE/code-judge && \
   MAX_EXECUTION_TIME=4 \
   REDIS_URI="redis://$MASTER_ADDR:6379" \
   RUN_WORKERS=0 \
   uvicorn app.main:app --host 0.0.0.0 --port 8088 --workers 16 \
   2>&1 | tee server.log'
```

#### 3. Start Code Judge Workers

```bash
# Launch workers (can be deployed on multiple nodes for increased parallelism)
# Adjust MAX_WORKERS based on your CPU count per node

tmux new-session -d -s worker \
  'cd $WORKSPACE/code-judge && \
   MAX_EXECUTION_TIME=4 \
   REDIS_URI="redis://$MASTER_ADDR:6379" \
   MAX_WORKERS=64 \
   python run_workers.py \
   2>&1 | tee worker.log'
```

### Data Preparation

This example uses:
- **Training Dataset**: DAPO-17k (English subset)
- **Test Dataset**: AIME24

```bash
# Process AIME 2024 dataset
python data_preprocess/aime2024_rstar2_agent_loop.py

# Process DAPO dataset
python data_preprocess/dapo_rstar2_agent_loop.py
```

### Model Setup

Download the base model (Qwen3-14B-Base):

```bash
huggingface-cli download Qwen/Qwen3-14B-Base --local-dir $HOME/models/Qwen3-14B-Base
```

> **Note**: The base model requires instruction-following SFT before RL training for optimal performance.

### Training

#### Basic Training

Run the training script (for 8x A100/H100 GPUs):

```bash
bash examples/run_qwen3-14b_rstar2_agent_weave.sh
```

> Adjust configuration parameters based on your hardware environment.

### Configuration

#### Data Augmentation Settings

The framework supports various sampling strategies to improve training efficiency:

```bash
# Global Settings
augmentation.do_down_sampling=True                                   # Enable down sampling
augmentation.down_sampling_config.down_sample_to_n=16                # Target number of traces per data point

# Sampling Strategies
augmentation.down_sampling_config.reject_equal_reward=True           # Enable reject sampling for equal rewards
augmentation.down_sampling_config.roc_error_ratio=True               # Resample correct traces by tool call error ratio
augmentation.down_sampling_config.roc_answer_format=True             # Resample correct traces by answer format

# Minimum Trace Requirements
augmentation.down_sampling_config.min_zero_reward_trace_num=2        # Minimum negative traces to retain
augmentation.down_sampling_config.min_non_zero_reward_trace_num=2    # Minimum positive traces to retain
```

### Troubleshooting

#### Common Issues

1. **Redis Connection Errors**: Ensure Redis is running and accessible at the specified address
2. **GPU Memory Issues**: Adjust batch sizes and model parameters for your hardware
3. **Code Judge Timeouts**: Increase `MAX_EXECUTION_TIME` for complex computations
4. **Worker Scaling**: Adjust `MAX_WORKERS` based on available CPU cores

#### Log Locations

- Server logs: `server.log` in the code-judge directory
- Worker logs: `worker.log` in the code-judge directory
- Training logs: Check your training script output directory

---


## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{guan2025rstar,
    title={rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking},
    author={Xinyu Guan and Li Lyna Zhang and Yifei Liu and Ning Shang and Youran Sun and Yi Zhu and Fan Yang and Mao Yang},
    year={2025},
    eprint={2501.04519},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
