import gymnasium as gym
import sinergym
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import datetime
import os
import json
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --------------------------------------------------------------------------------------
# 📅 Timestamp, Env Setup
# --------------------------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
experiment_name = f'PPO-5ZoneHot-NormTrain-{timestamp}'
env_id = 'Eplus-5zone-hot-continuous-v1'

extra_conf = {
    'timesteps_per_hour': 1,
    'runperiod': (1, 1, 1991, 31, 12, 1991),
    'reward': reward_kwargs
}

# ---------------------------------------------------------------------------------------
# 🧠 DistilBERT Feature Extractor for PPO
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
        
        logger.info(f"DistilBERT Feature Extractor initialized on {self.device}")

    def forward(self, observations):
        # Handle different observation formats
        if isinstance(observations, dict):
            obs = observations['obs']
        else:
            obs = observations
            
        # Convert observations to text format for DistilBERT
        if isinstance(obs, np.ndarray):
            # Convert numerical observations to descriptive text
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
# 🏗️ Custom Policy Network for PPO
# ---------------------------------------------------------------------------------------
class CustomPolicyNetwork(nn.Module):
    def __init__(self, observation_dim=256, action_dim=2, hidden_dim=128):
        super().__init__()
        
        # Policy head (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.policy_net(x), self.value_net(x)

# ---------------------------------------------------------------------------------------
# 🔧 Environment Wrapper for Observation Processing
# ---------------------------------------------------------------------------------------
class SinergymObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Keep original observation space for numerical processing
        self.observation_space = env.observation_space

    def observation(self, obs):
        # Ensure observations are properly formatted
        if isinstance(obs, dict):
            # Convert dict to array if needed
            obs_array = np.array(list(obs.values()), dtype=np.float32)
        elif isinstance(obs, np.ndarray):
            obs_array = obs.astype(np.float32)
        else:
            obs_array = np.array(obs, dtype=np.float32)
        
        # Handle NaN and Inf values
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return obs_array

# ---------------------------------------------------------------------------------------
# 📊 Training Configuration
# ---------------------------------------------------------------------------------------
def get_training_config():
    return {
        'total_timesteps': 1000000,  # 1M timesteps
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 2048,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'target_kl': 0.01,
        'tensorboard_log': f"./tensorboard_logs/{experiment_name}/",
        'verbose': 1
    }

# ---------------------------------------------------------------------------------------
# 🚀 Main Training Function
# ---------------------------------------------------------------------------------------
def train_ppo_distilbert():
    logger.info("Starting PPO training with DistilBERT for Sinergym")
    
    # Create environment
    env = gym.make(env_id, **extra_conf)
    env = SinergymObsWrapper(env)
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create evaluation environment
    eval_env = gym.make(env_id, **extra_conf)
    eval_env = SinergymObsWrapper(eval_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    eval_vec_env = VecNormalize(
        eval_vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Policy configuration
    policy_kwargs = dict(
        features_extractor_class=DistilBertFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[128, 128],  # Policy network
            vf=[128, 128]   # Value network
        ),
        activation_fn=nn.ReLU
    )
    
    # Training configuration
    config = get_training_config()
    
    # Create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        **config
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=f"./models/{experiment_name}/",
        log_path=f"./logs/{experiment_name}/",
        eval_freq=max(10000 // config['n_steps'], 1),
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // config['n_steps'], 1),
        save_path=f"./checkpoints/{experiment_name}/",
        name_prefix="ppo_distilbert"
    )
    
    # Create directories
    os.makedirs(f"./models/{experiment_name}/", exist_ok=True)
    os.makedirs(f"./logs/{experiment_name}/", exist_ok=True)
    os.makedirs(f"./checkpoints/{experiment_name}/", exist_ok=True)
    
    # Save configuration
    with open(f"./logs/{experiment_name}/config.json", 'w') as f:
        json.dump({
            'env_id': env_id,
            'extra_conf': extra_conf,
            'reward_kwargs': reward_kwargs,
            'training_config': config,
            'policy_kwargs': policy_kwargs
        }, f, indent=2)
    
    logger.info(f"Starting training for {config['total_timesteps']} timesteps")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Train the model
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save final model
    model.save(f"./models/{experiment_name}/final_model")
    
    # Save environment normalization
    vec_env.save(f"./models/{experiment_name}/vec_normalize.pkl")
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: ./models/{experiment_name}/")
    
    return model, vec_env

# ---------------------------------------------------------------------------------------
# 🧪 Evaluation Function
# ---------------------------------------------------------------------------------------
def evaluate_model(model_path, env_id, num_episodes=10):
    """Evaluate a trained model"""
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make(env_id, **extra_conf)
    env = SinergymObsWrapper(env)
    
    # Load normalization if available
    vec_normalize_path = model_path.replace('.zip', '_vec_normalize.pkl')
    if os.path.exists(vec_normalize_path):
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Average Episode Length: {avg_length:.2f}")
    
    return episode_rewards, episode_lengths

# ---------------------------------------------------------------------------------------
# 🎯 Main Execution
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Train model
    model, vec_env = train_ppo_distilbert()
    
    # Evaluate best model
    best_model_path = f"./models/{experiment_name}/best_model"
    if os.path.exists(best_model_path + ".zip"):
        logger.info("Evaluating best model...")
        evaluate_model(best_model_path, env_id)
    
    logger.info("All done!")