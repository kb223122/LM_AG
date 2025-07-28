import gymnasium as gym
import sinergym
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# 1. DistilBERT Feature Extractor for PPO
class DistilBertFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, model_name='distilbert-base-uncased', features_dim=128):
        super().__init__(observation_space, features_dim)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.bert.eval()  # Set to eval mode
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT
        self.fc = nn.Linear(self.bert.config.hidden_size, features_dim)

    def forward(self, observations):
        # observations: batch of strings
        if isinstance(observations, dict):
            obs = observations['obs']
        else:
            obs = observations
        if isinstance(obs, np.ndarray):
            obs = obs.tolist()
        if isinstance(obs, list) and isinstance(obs[0], bytes):
            obs = [o.decode('utf-8') for o in obs]
        elif isinstance(obs, bytes):
            obs = [obs.decode('utf-8')]
        elif isinstance(obs, str):
            obs = [obs]
        # Tokenize
        tokens = self.tokenizer(obs, padding=True, truncation=True, max_length=32, return_tensors='pt')
        tokens = {k: v.to(self.fc.weight.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.bert(**tokens)
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.fc(pooled)

# 2. Sinergym Environment Wrapper (for string state)
class SinergymStringObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Replace observation space with a dummy Box (SB3 expects Box)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def observation(self, obs):
        # Convert dict/array obs to string for DistilBERT
        if isinstance(obs, dict):
            obs_str = ' '.join([str(v) for v in obs.values()])
        elif isinstance(obs, np.ndarray):
            obs_str = ' '.join([str(x) for x in obs])
        else:
            obs_str = str(obs)
        return np.array([obs_str], dtype=object)

# 3. Main training script
def main():
    env_id = 'Eplus-office-hot-discrete-v1'  # Default Sinergym env
    env = gym.make(env_id)
    env = SinergymStringObsWrapper(env)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=DistilBertFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_sinergym_tensorboard/"
    )

    model.learn(total_timesteps=10000)  # Adjust as needed
    model.save("ppo_distilbert_sinergym")
    print("Model saved as ppo_distilbert_sinergym.zip")

if __name__ == "__main__":
    main()