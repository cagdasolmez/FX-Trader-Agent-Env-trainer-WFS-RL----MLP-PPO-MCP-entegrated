# FX-Trader-Agent-Env-trainer-WFS-RL----MLP-PPO-MCP-entegrated
In this project, I developed a deep reinforcement learning (RL) framework using PyTorch based on a Multi-Layer Perceptron (MLP) Actor-Critic architecture trained with the Proximal Policy Optimization (PPO) algorithm. For data partitioning, I designed a rolling window scheme in which 10 candlesticks were used for training and 1 candlestick for testing/validation. By shifting this window forward step-by-step, I implemented a walk-forward training procedure to better reflect real-world sequential learning dynamics.

For anomaly detection, I employed a rolling Z-score method to identify outliers, with a particular focus on flagging price spikes occurring in low-volume regimes. These flagged regions were integrated into the learning process to improve the agent’s sensitivity to unusual market behavior.

On the environment side, I mathematically formalized well-known expert-defined candlestick patterns and modeled the signal generation process as a pattern-matching mechanism against these atomic structures. The training objective was designed around this formulation to align the agent’s decision-making with interpretable market signals.

I constructed an MLP-based actor-critic encoder for the agent and introduced a gating mechanism to enhance responsiveness in outlier-flagged regions. To enable interaction between the agent and the external market environment, I built an MCP (Market Communication Protocol) module. Additionally, I developed a standardized PyTorch training pipeline and performed systematic fine-tuning of the model.

After training, I evaluated model performance using multiple decision strategies, including greedy, stochastic, temperature-based, and epsilon-based testing layers. For each evaluation mode, I computed confusion matrices to quantitatively assess prediction quality and analyze behavioral trade-offs across different inference settings.
