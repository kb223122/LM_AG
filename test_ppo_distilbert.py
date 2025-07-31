#!/usr/bin/env python3
"""
Test script for PPO + DistilBERT implementation

This script runs a quick test to verify the implementation works correctly.
"""

import sys
import os
import torch
import numpy as np
import gymnasium as gym
import sinergym

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    """Test environment creation and basic functionality"""
    print("🧪 Testing environment creation...")
    
    try:
        # Import from main script
        from ppo_distilbert_training import (
            env_id, extra_conf, reward_kwargs,
            DistilBertFeatureExtractor
        )
        
        # Create environment
        train_env = gym.make(env_id, config_params=extra_conf, env_name='test-env')
        
        # Test reset
        obs, info = train_env.reset()
        print(f"✅ Environment created successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation type: {type(obs)}")
        print(f"   Observation space: {train_env.observation_space}")
        print(f"   Action space: {train_env.action_space}")
        
        # Test step
        action = train_env.action_space.sample()
        obs, reward, done, truncated, info = train_env.step(action)
        print(f"✅ Step successful")
        print(f"   Action: {action}")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        
        train_env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_distilbert_extractor():
    """Test DistilBERT feature extractor"""
    print("\n🧪 Testing DistilBERT feature extractor...")
    
    try:
        from ppo_distilbert_training import DistilBertFeatureExtractor
        
        # Create mock observation space
        from gymnasium.spaces import Box
        obs_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        # Create feature extractor
        extractor = DistilBertFeatureExtractor(obs_space, features_dim=256)
        
        # Create mock observations
        mock_obs = np.random.randn(2, 17).astype(np.float32)  # Batch of 2 observations
        
        # Test forward pass
        features = extractor(mock_obs)
        print(f"✅ DistilBERT feature extractor created successfully")
        print(f"   Input shape: {mock_obs.shape}")
        print(f"   Output shape: {features.shape}")
        print(f"   Device: {extractor.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ DistilBERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_small_training():
    """Test small training run"""
    print("\n🧪 Testing small training run...")
    
    try:
        from stable_baselines3 import PPO
        from ppo_distilbert_training import (
            env_id, extra_conf, reward_kwargs,
            DistilBertFeatureExtractor
        )
        from sinergym.utils.wrappers import (
            LoggerWrapper, CSVLogger, NormalizeObservation, NormalizeAction
        )
        from sinergym.utils.rewards import LinearReward
        
        # Create environment
        train_env = gym.make(env_id, config_params=extra_conf, env_name='test-train')
        train_env = NormalizeObservation(train_env)
        train_env = NormalizeAction(train_env)
        train_env = LoggerWrapper(train_env)
        train_env = CSVLogger(train_env)
        train_env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))
        
        # Policy configuration
        policy_kwargs = dict(
            features_extractor_class=DistilBertFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
        
        # Create model
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            verbose=0
        )
        
        # Train for a few steps
        print("   Training for 5000 timesteps...")
        model.learn(total_timesteps=5000)
        
        print("✅ Small training test successful")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 PPO + DistilBERT Implementation Test")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Run tests
    tests = [
        ("Environment", test_environment),
        ("DistilBERT Extractor", test_distilbert_extractor),
        ("Small Training", test_small_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Implementation is ready to use.")
        print("\nTo start training, run:")
        print("python ppo_distilbert_training.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()