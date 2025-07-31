#!/usr/bin/env python3
"""
Test script for the fixed PPO + DistilBERT implementation
"""

import sys
import os
import torch
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_distilbert_extractor():
    """Test DistilBERT feature extractor"""
    print("🧪 Testing DistilBERT feature extractor...")
    
    try:
        from ppo_distilbert_training_fixed import DistilBertFeatureExtractor
        
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

def test_environment_creation():
    """Test environment creation"""
    print("\n🧪 Testing environment creation...")
    
    try:
        import gymnasium as gym
        import sinergym
        from ppo_distilbert_training_fixed import env_id, extra_conf, reward_kwargs
        
        # Try with config_params
        try:
            env = gym.make(env_id, config_params=extra_conf, env_name='test-env')
            print("✅ Environment created with config_params")
        except Exception as e:
            print(f"⚠️  config_params failed: {e}")
            # Try without config_params
            env = gym.make(env_id, env_name='test-env')
            print("✅ Environment created without config_params")
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_space}")
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"   Step successful, reward: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Run tests"""
    print("=" * 60)
    print("🧪 Fixed PPO + DistilBERT Implementation Test")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Run tests
    tests = [
        ("Environment Creation", test_environment_creation),
        ("DistilBERT Extractor", test_distilbert_extractor),
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
        print("🎉 All tests passed! Fixed implementation is ready to use.")
        print("\nTo start training, run:")
        print("python ppo_distilbert_training_fixed.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()