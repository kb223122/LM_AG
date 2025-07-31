# Sinergym DistilBERT PPO Implementation

This repository implements a PPO (Proximal Policy Optimization) agent with DistilBERT feature extraction for building energy management using the Sinergym environment. This work is inspired by the LM_AG crop management paper but adapted for building energy optimization.

## 🎯 Overview

The implementation combines:
- **PPO Algorithm**: For stable policy optimization
- **DistilBERT**: For advanced feature extraction from building state observations
- **Sinergym**: EnergyPlus-based building simulation environment
- **Custom Reward Function**: Balancing comfort and energy efficiency

## 🏗️ Architecture

### Observation Space
- **17 variables**: Building environment observations including temperature, humidity, CO2, occupancy, HVAC power, etc.
- **Shape**: (17,) float32 array
- **Range**: [-5e7, 5e7] (normalized during training)

### Action Space
- **2 continuous actions**: Heating and cooling setpoints
- **Shape**: (2,) float32 array
- **Control**: Continuous HVAC setpoint management

### Neural Network Architecture
```
DistilBERT Feature Extractor:
├── DistilBERT (frozen) → 768-dim embeddings
├── Linear projection → 256-dim features
└── ReLU + Dropout

Policy Network (Actor):
├── Input: 256-dim features
├── Hidden: 128 → 128 (ReLU + Dropout)
└── Output: 2-dim actions

Value Network (Critic):
├── Input: 256-dim features
├── Hidden: 128 → 128 (ReLU + Dropout)
└── Output: 1-dim value
```

## ⚙️ Configuration

### Environment Setup
```python
env_id = 'Eplus-5zone-hot-continuous-v1'
extra_conf = {
    'timesteps_per_hour': 1,
    'runperiod': (1, 1, 1991, 31, 12, 1991),  # Full year 1991
    'reward': reward_kwargs
}
```

### Reward Function
```python
reward_kwargs = {
    "temperature_variables": ["air_temperature"],
    "energy_variables": ["HVAC_electricity_demand_rate"],
    "range_comfort_winter": [20.0, 23.5],
    "range_comfort_summer": [23.0, 26.0],
    "summer_start": [6, 1],
    "summer_final": [9, 30],
    "energy_weight": 0.35,
    "lambda_energy": 0.01,
    "lambda_temperature": 28
}
```

### Training Parameters
```python
config = {
    'total_timesteps': 1000000,  # 1M timesteps
    'learning_rate': 3e-4,
    'batch_size': 64,
    'n_steps': 2048,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'target_kl': 0.01
}
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train_sinergym_distilbert.py
```

### 3. Monitor Training
```bash
tensorboard --logdir ./tensorboard_logs/
```

## 📁 Project Structure

```
├── sinergym_distilbert_ppo_complete.py  # Main implementation
├── train_sinergym_distilbert.py         # Training script
├── requirements.txt                      # Dependencies
├── README_Sinergym_DistilBERT.md        # This file
├── models/                              # Saved models (generated)
│   └── PPO-5ZoneHot-NormTrain-YYYYMMDD-HHMMSS/
├── logs/                                # Training logs (generated)
│   └── PPO-5ZoneHot-NormTrain-YYYYMMDD-HHMMSS/
├── checkpoints/                         # Training checkpoints (generated)
│   └── PPO-5ZoneHot-NormTrain-YYYYMMDD-HHMMSS/
└── tensorboard_logs/                    # TensorBoard logs (generated)
    └── PPO-5ZoneHot-NormTrain-YYYYMMDD-HHMMSS/
```

## 🔧 Key Features

### 1. DistilBERT Feature Extraction
- Converts numerical observations to descriptive text
- Uses pre-trained DistilBERT for semantic understanding
- Freezes BERT parameters for efficiency
- Projects to 256-dimensional features

### 2. Observation Processing
- Handles 17 building environment variables
- Converts to descriptive text for BERT
- Normalizes and handles NaN/Inf values
- Supports both dict and array observations

### 3. Efficient Training
- GPU acceleration when available
- Observation and reward normalization
- Automatic checkpointing and evaluation
- TensorBoard logging

### 4. Custom Reward Function
- Balances thermal comfort and energy efficiency
- Seasonal comfort ranges (winter/summer)
- Configurable energy and temperature weights
- Penalizes comfort violations and energy consumption

## 📊 Expected Results

The agent should learn to:
- Maintain comfortable indoor temperatures (20-26°C)
- Minimize HVAC energy consumption
- Adapt to seasonal changes
- Handle occupancy patterns
- Optimize setpoints for energy efficiency

## 🧪 Evaluation

The training includes automatic evaluation with:
- Best model saving based on evaluation performance
- Periodic evaluation during training
- Final model evaluation with multiple episodes
- Detailed logging of rewards and episode lengths

## 🔍 Monitoring

### TensorBoard Metrics
- `train/episode_reward_mean`: Average training reward
- `train/episode_length_mean`: Average episode length
- `eval/episode_reward_mean`: Average evaluation reward
- `train/entropy_loss`: Policy entropy
- `train/value_loss`: Value function loss
- `train/policy_loss`: Policy gradient loss

### Log Files
- `config.json`: Complete training configuration
- `evaluations.npz`: Evaluation results
- `PPO_*.log`: Training logs

## 🛠️ Customization

### Modify Reward Function
Edit the `reward_kwargs` in `sinergym_distilbert_ppo_complete.py`:
```python
reward_kwargs = {
    "energy_weight": 0.5,  # Increase energy penalty
    "lambda_temperature": 30,  # Adjust comfort penalty
    # ... other parameters
}
```

### Change Environment
Modify `env_id` and `extra_conf`:
```python
env_id = 'Eplus-office-hot-continuous-v1'  # Different building
extra_conf = {
    'runperiod': (1, 1, 1991, 31, 3, 1991),  # Different period
    # ... other config
}
```

### Adjust Training Parameters
Modify `get_training_config()`:
```python
def get_training_config():
    return {
        'total_timesteps': 2000000,  # More training
        'learning_rate': 1e-4,       # Different LR
        # ... other parameters
    }
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or n_steps
   - Use CPU training if GPU memory is limited

2. **Training Instability**
   - Adjust learning rate
   - Modify reward weights
   - Increase entropy coefficient

3. **Poor Performance**
   - Increase training timesteps
   - Adjust network architecture
   - Fine-tune reward function

### Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Memory**: Monitor GPU memory usage during training
3. **Checkpoints**: Use checkpoints to resume interrupted training
4. **Evaluation**: Monitor evaluation metrics to detect overfitting

## 📚 References

- [Sinergym Documentation](https://ugr-sail.github.io/sinergym/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.