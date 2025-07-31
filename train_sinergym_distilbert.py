#!/usr/bin/env python3
"""
Sinergym DistilBERT PPO Training Script

This script trains a PPO agent with DistilBERT feature extraction for building energy management
using the Sinergym environment.

Usage:
    python train_sinergym_distilbert.py
"""

import sys
import os
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sinergym_distilbert_ppo_complete import train_ppo_distilbert, evaluate_model, experiment_name, env_id

def main():
    """Main training function"""
    print("=" * 80)
    print("🚀 Sinergym DistilBERT PPO Training")
    print("=" * 80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    print(f"📊 Environment: {env_id}")
    print(f"🏷️  Experiment: {experiment_name}")
    print("=" * 80)
    
    try:
        # Train the model
        model, vec_env = train_ppo_distilbert()
        
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        
        # Evaluate the best model
        best_model_path = f"./models/{experiment_name}/best_model"
        if os.path.exists(best_model_path + ".zip"):
            print("\n🧪 Evaluating best model...")
            evaluate_model(best_model_path, env_id, num_episodes=5)
        
        print("\n🎉 All done! Check the logs and models directories for results.")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()