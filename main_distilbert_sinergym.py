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
from stable_baselines3.common.policies import ActorCriticPolicy
from transformers import DistilBertTokenizerFast, DistilBertModel
from gymnasium import spaces
import torch
import torch.nn as nn
import numpy as np
import os
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import time
import csv

# ======== Tokenizer =========
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

PROFILE_CSV = "profiling_log.csv"
def log_profile(component, duration):
    with open(PROFILE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([component, duration])

# ======== Numeric String Obs Wrapper with Profiling =========
class NumericStringObservationWrapper(gym.ObservationWrapper):
    """
    Converts a float32 Box(17,) observation to a string, tokenizes, and outputs token ids (int32, shape [32]).
    """
    def __init__(self, env):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.observation_space = spaces.Box(
            low=0, high=tokenizer.vocab_size,
            shape=(32,), dtype=np.int32
        )

    def observation(self, obs):
        t0 = time.time()
        obs_str = ' '.join([str(round(float(x), 2)) for x in obs])
        t1 = time.time()
        encoded = self.tokenizer(
            obs_str,
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors="np"
        )
        t2 = time.time()
        log_profile("obs_to_string", t1-t0)
        log_profile("tokenization", t2-t1)
        return encoded["input_ids"].squeeze(0)

# ======== DistilBERT Extractor with Profiling =========
class DistilBertExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert.to(device)
        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim)
        ).to(device)

    def forward(self, obs):
        device = next(self.linear.parameters()).device
        t0 = time.time()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.long().to(device)
        t1 = time.time()
        attention_mask = (obs != tokenizer.pad_token_id).long().to(device)
        t2 = time.time()
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        t3 = time.time()
        log_profile("extractor_obs_to_tensor", t1-t0)
        log_profile("attention_mask", t2-t1)
        log_profile("attention_mask_dim", t3-t2)
        t4 = time.time()
        outputs = self.bert(input_ids=obs, attention_mask=attention_mask)
        t5 = time.time()
        cls_embedding = outputs.last_hidden_state[:, 0]
        t6 = time.time()
        out = self.linear(cls_embedding)
        t7 = time.time()
        log_profile("bert_forward", t5-t4)
        log_profile("cls_slice", t6-t5)
        log_profile("linear_head", t7-t6)
        return out

# ======== Custom Policy =========
class PPODistilBertPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=DistilBertExtractor,
                         features_extractor_kwargs=dict(features_dim=128))

# ======== Reward Setup =========
lambda_temperature = 28
lambda_energy = 0.01
energy_weight = 0.4
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

# ======== Environment Setup =========
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
experiment_name = f'PPO-DistilBert-numericstring-{timestamp}'
env_id = 'Eplus-5zone-hot-continuous-v1'

# --- Training Environment ---
train_env = gym.make(env_id)
train_env = LoggerWrapper(train_env)
train_env = CSVLogger(train_env)
train_env = NormalizeObservation(train_env)
train_env = NormalizeAction(train_env)
train_env = NumericStringObservationWrapper(train_env)
train_env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))

# --- Timing Wrapper for Profiling ---
class TimingWrapper(gym.Wrapper):
    def step(self, action):
        t0 = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        t1 = time.time()
        log_profile("sinergym_step", t1-t0)
        return obs, reward, terminated, truncated, info

train_env = TimingWrapper(train_env)

# --- Evaluation Environment ---
eval_env = gym.make(env_id)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = NumericStringObservationWrapper(eval_env)
eval_env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))

# ======== PPO Model with Profiling Callback =========
class ProfilingCallback(CallbackList):
    def __init__(self, callbacks=None):
        super().__init__(callbacks or [])
        self.ppo_step_times = []

    def _on_step(self) -> bool:
        t0 = time.time()
        result = super()._on_step()
        t1 = time.time()
        log_profile("ppo_training_step", t1-t0)
        return result

# ======== Callback =========
eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=train_env,
    n_eval_episodes=3,
    eval_freq_episodes=2,
    deterministic=True
)
callback = ProfilingCallback([eval_callback])

# ======== Train =========
episodes = 3  # For real training, use a higher number (e.g., 500)
total_timesteps = episodes * train_env.get_wrapper_attr('timestep_per_episode')
with open(PROFILE_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["component", "duration_sec"])
model = PPO(
    policy=PPODistilBertPolicy,
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    verbose=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=500)

# ======== Save Model =========
workspace = train_env.get_wrapper_attr('workspace_path')
os.makedirs(workspace, exist_ok=True)
model_path = os.path.join(workspace, f'ppo_distilbert_{timestamp}')
model.save(model_path)
with open(os.path.join(workspace, f'reward_{timestamp}.json'), 'w') as f:
    json.dump(reward_kwargs, f, indent=4)

# ======== Plot Reward Curve =========
progress_path = os.path.join(workspace, 'progress.csv')
progress_df = pd.read_csv(progress_path)
plt.figure(figsize=(8, 6))
plt.plot(progress_df['episode_num'], progress_df['mean_reward'], marker='o')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('PPO + DistilBERT (Numeric String) Reward Convergence')
plt.grid(True)
plt.savefig(os.path.join(workspace, f'reward_plot_{timestamp}.png'))
plt.show()

print(f"\nProfiling results saved to {PROFILE_CSV}")