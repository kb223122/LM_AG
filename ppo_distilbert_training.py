# ---------------------------------------------------------------------------------------
# ✅ PPO + DistilBERT Training Script (Sinergym 3.7.3)
# ---------------------------------------------------------------------------------------

import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import (
    LoggerWrapper, CSVLogger, NormalizeObservation, NormalizeAction
)
from sinergym.utils.rewards import LinearReward
from sinergym.utils.callbacks import LoggerEvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import json
import numpy as np
import time
from typing import Dict, Any

# ---------------------------------------------------------------------------------------
# ⚙️ Custom Reward Parameters
# ---------------------------------------------------------------------------------------
lambda_temperature = 28
lambda_energy = 0.01
energy_weight = 0.35

reward_kwargs = {
    "temperature_variables": ["air_temperature"],
    "energy_variables": ["HVAC_electricity_demand_rate"],
    "range_comfort_winter": [20.0, 23.5],
    "range_comfort_summer": [23.0, 26.0],
    "summer_start": [6, 1],
    "summer_final": [9, 30],
    "energy_weight": energy_weight,
    "lambda_energy": lambda_energy,
    "lambda_temperature": lambda_temperature
}

# ---------------------------------------------------------------------------------------
# 🧠 DistilBERT Feature Extractor
# ---------------------------------------------------------------------------------------
class DistilBertFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, model_name='distilbert-base-uncased', features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Initialize DistilBERT
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters for efficiency
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Feature projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert.to(self.device)
        self.feature_projection.to(self.device)
        
        print(f"✅ DistilBERT Feature Extractor initialized on {self.device}")

    def forward(self, observations):
        # Handle different observation formats
        if isinstance(observations, dict):
            obs = observations['obs']
        else:
            obs = observations
            
        # Convert observations to text format for DistilBERT
        if isinstance(obs, np.ndarray):
            obs_texts = self._convert_obs_to_text(obs)
        else:
            obs_texts = [str(obs)]
        
        # Tokenize
        tokens = self.tokenizer(
            obs_texts, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors='pt'
        )
        
        # Move tokens to device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**tokens)
            # Use [CLS] token representation
            pooled = outputs.last_hidden_state[:, 0, :]
        
        # Project to feature dimension
        features = self.feature_projection(pooled)
        return features
    
    def _convert_obs_to_text(self, obs_array):
        """Convert numerical observations to descriptive text for DistilBERT"""
        texts = []
        
        # Sinergym observation variables (17 total)
        obs_names = [
            "outdoor_temperature", "outdoor_humidity", "outdoor_pressure", "outdoor_wind_speed",
            "outdoor_wind_direction", "diffuse_solar_radiation", "direct_solar_radiation",
            "zone_1_temperature", "zone_1_humidity", "zone_1_co2", "zone_1_occupancy",
            "zone_2_temperature", "zone_2_humidity", "zone_2_co2", "zone_2_occupancy",
            "hvac_power", "hvac_energy"
        ]
        
        for obs in obs_array:
            text_parts = []
            for i, (name, value) in enumerate(zip(obs_names, obs)):
                if not np.isnan(value) and not np.isinf(value):
                    text_parts.append(f"{name}: {value:.2f}")
            
            # Create descriptive text
            text = "Building environment: " + ", ".join(text_parts)
            texts.append(text)
        
        return texts

# ---------------------------------------------------------------------------------------
# 📅 Timestamp, Env Setup
# ---------------------------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
experiment_name = f'PPO-DistilBERT-5ZoneHot-{timestamp}'
env_id = 'Eplus-5zone-hot-continuous-v1'

extra_conf = {
    'timesteps_per_hour': 1,
    'runperiod': (1, 1, 1991, 31, 12, 1991),
    'reward': reward_kwargs
}

# ---------------------------------------------------------------------------------------
# 🕐 Time Profiling Setup
# ---------------------------------------------------------------------------------------
class TimeProfiler:
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, name):
        self.start_times[name] = time.time()
    
    def end(self, name):
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            del self.start_times[name]
    
    def get_stats(self, name):
        if name in self.timings:
            times = self.timings[name]
            return {
                'count': len(times),
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times)
            }
        return None
    
    def print_summary(self):
        print("\n" + "="*60)
        print("⏱️  TIME PROFILING SUMMARY")
        print("="*60)
        for name in self.timings:
            stats = self.get_stats(name)
            if stats:
                print(f"{name:25} | Count: {stats['count']:4d} | "
                      f"Total: {stats['total']:8.2f}s | "
                      f"Mean: {stats['mean']:6.2f}s | "
                      f"Std: {stats['std']:6.2f}s")
        print("="*60)

profiler = TimeProfiler()

# ---------------------------------------------------------------------------------------
# 🌱 Training and Evaluation Environment
# ---------------------------------------------------------------------------------------
print("🚀 Setting up environments...")
profiler.start('env_setup')

train_env = gym.make(env_id, config_params=extra_conf, env_name=experiment_name)
train_env = NormalizeObservation(train_env)
train_env = NormalizeAction(train_env)
train_env = LoggerWrapper(train_env)
train_env = CSVLogger(train_env)
train_env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))

eval_env = gym.make(env_id, config_params=extra_conf, env_name=experiment_name + '-EVAL')
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
eval_env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))

profiler.end('env_setup')

# ---------------------------------------------------------------------------------------
# 🤖 PPO Agent with DistilBERT Architecture
# ---------------------------------------------------------------------------------------
print("🤖 Creating PPO agent with DistilBERT...")
profiler.start('model_creation')

# Policy configuration with DistilBERT
policy_kwargs = dict(
    features_extractor_class=DistilBertFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(
        pi=[128, 128],  # Policy network
        vf=[128, 128]   # Value network
    ),
    activation_fn=nn.ReLU
)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = PPO(
    policy='MlpPolicy',
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    verbose=1,
    device=device,
    policy_kwargs=policy_kwargs
)

profiler.end('model_creation')

# ---------------------------------------------------------------------------------------
# 📈 Callbacks for Logging
# ---------------------------------------------------------------------------------------
print("📈 Setting up callbacks...")
profiler.start('callback_setup')

eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=train_env,
    n_eval_episodes=3,
    eval_freq_episodes=10,
    deterministic=True
)
callback = CallbackList([eval_callback])

profiler.end('callback_setup')

# ---------------------------------------------------------------------------------------
# ▶️ Train the Agent
# ---------------------------------------------------------------------------------------
episodes = 5000
timesteps_per_episode = train_env.get_wrapper_attr('timestep_per_episode')
total_timesteps = episodes * timesteps_per_episode

print(f"\n🚀 Training PPO + DistilBERT for {total_timesteps} timesteps ({episodes} episodes)...")
print(f"   Lambda Temperature: {lambda_temperature}")
print(f"   Lambda Energy: {lambda_energy}")
print(f"   Energy Weight: {energy_weight}")
print(f"   Device: {device}")
print()

profiler.start('total_training')

model.learn(
    total_timesteps=total_timesteps,
    callback=callback,
    log_interval=500
)

profiler.end('total_training')

# ---------------------------------------------------------------------------------------
# 💾 Save Model and Reward Config
# ---------------------------------------------------------------------------------------
print("\n💾 Saving model and configuration...")
profiler.start('model_saving')

workspace = train_env.get_wrapper_attr('workspace_path')
os.makedirs(workspace, exist_ok=True)

model_filename = f'ppo_distilbert_Lt{lambda_temperature}_Le{lambda_energy}_w{energy_weight}_{timestamp}'
model_path = os.path.join(workspace, model_filename)
model.save(model_path)
print(f"✅ Model saved to: {model_path}.zip")

reward_settings_path = os.path.join(workspace, f'reward_settings_{timestamp}.json')
with open(reward_settings_path, 'w') as f:
    json.dump(reward_kwargs, f, indent=4)
print(f"✅ Reward settings saved to: {reward_settings_path}")

profiler.end('model_saving')

# ---------------------------------------------------------------------------------------
# 📊 Plot Training Reward Convergence
# ---------------------------------------------------------------------------------------
print("\n📊 Generating reward convergence plot...")
profiler.start('plotting')

progress_path = os.path.join(workspace, 'progress.csv')
if os.path.exists(progress_path):
    progress_df = pd.read_csv(progress_path)
    
    plt.figure(figsize=(12, 8))
    
    # Main reward plot
    plt.subplot(2, 2, 1)
    plt.plot(progress_df['episode_num'], progress_df['mean_reward'], 
             marker='o', color='blue', alpha=0.7, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(f'PPO + DistilBERT Reward Convergence\n(Lt={lambda_temperature}, Le={lambda_energy}, w={energy_weight})')
    plt.grid(True, alpha=0.3)
    
    # Moving average
    plt.subplot(2, 2, 2)
    window_size = 50
    if len(progress_df) > window_size:
        moving_avg = progress_df['mean_reward'].rolling(window=window_size).mean()
        plt.plot(progress_df['episode_num'], moving_avg, 
                color='red', linewidth=2, label=f'{window_size}-episode moving average')
        plt.plot(progress_df['episode_num'], progress_df['mean_reward'], 
                color='blue', alpha=0.3, label='Raw rewards')
        plt.legend()
    else:
        plt.plot(progress_df['episode_num'], progress_df['mean_reward'], 
                color='blue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward (Moving Average)')
    plt.title('Reward Convergence with Moving Average')
    plt.grid(True, alpha=0.3)
    
    # Episode length
    plt.subplot(2, 2, 3)
    plt.plot(progress_df['episode_num'], progress_df['episode_length'], 
             marker='s', color='green', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Over Time')
    plt.grid(True, alpha=0.3)
    
    # Training statistics
    plt.subplot(2, 2, 4)
    if 'value_loss' in progress_df.columns:
        plt.plot(progress_df['episode_num'], progress_df['value_loss'], 
                color='orange', alpha=0.7, label='Value Loss')
    if 'policy_loss' in progress_df.columns:
        plt.plot(progress_df['episode_num'], progress_df['policy_loss'], 
                color='purple', alpha=0.7, label='Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(workspace, f'PPO_DistilBERT_Convergence_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Reward plot saved to: {plot_path}")
else:
    print("⚠️  Progress file not found, skipping plot generation")

profiler.end('plotting')

# ---------------------------------------------------------------------------------------
# 📋 Print Final Statistics
# ---------------------------------------------------------------------------------------
print("\n" + "="*60)
print("📋 TRAINING SUMMARY")
print("="*60)
print(f"Experiment Name: {experiment_name}")
print(f"Environment: {env_id}")
print(f"Total Episodes: {episodes}")
print(f"Total Timesteps: {total_timesteps}")
print(f"Device Used: {device}")
print(f"Model Saved: {model_path}.zip")

if os.path.exists(progress_path):
    progress_df = pd.read_csv(progress_path)
    final_reward = progress_df['mean_reward'].iloc[-1]
    best_reward = progress_df['mean_reward'].max()
    avg_reward = progress_df['mean_reward'].mean()
    print(f"Final Reward: {final_reward:.2f}")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")

# Print time profiling summary
profiler.print_summary()

print("\n🎉 Training completed successfully!")
print("="*60)