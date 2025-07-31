#!/usr/bin/env python3
"""
Model Evaluation Script for Sinergym DistilBERT PPO

This script evaluates a trained model and provides detailed performance metrics.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sinergym_distilbert_ppo_complete import (
    evaluate_model, 
    env_id, 
    extra_conf,
    SinergymObsWrapper
)

def evaluate_model_detailed(model_path, num_episodes=20, save_results=True):
    """Detailed model evaluation with comprehensive metrics"""
    
    print(f"🔍 Evaluating model: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model not found: {model_path}.zip")
        return None
    
    try:
        # Load model
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        
        # Create environment
        import gymnasium as gym
        import sinergym
        env = gym.make(env_id, **extra_conf)
        env = SinergymObsWrapper(env)
        
        # Load normalization if available
        vec_normalize_path = model_path.replace('.zip', '_vec_normalize.pkl')
        if os.path.exists(vec_normalize_path):
            from stable_baselines3.common.vec_env import VecNormalize
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            print("✅ Loaded environment normalization")
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        episode_actions = []
        episode_observations = []
        
        print(f"🧪 Running {num_episodes} evaluation episodes...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_action_list = []
            episode_obs_list = []
            
            while True:
                # Store observation
                episode_obs_list.append(obs.copy())
                
                # Get action
                action, _ = model.predict(obs, deterministic=True)
                episode_action_list.append(action.copy())
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_actions.append(np.array(episode_action_list))
            episode_observations.append(np.array(episode_obs_list))
            
            print(f"   Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}, Length = {episode_length:3d}")
        
        # Calculate statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        avg_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        print(f"Average Reward:     {avg_reward:8.2f} ± {std_reward:6.2f}")
        print(f"Reward Range:       {min_reward:8.2f} to {max_reward:8.2f}")
        print(f"Average Length:     {avg_length:8.2f} ± {std_length:6.2f}")
        print(f"Total Episodes:     {num_episodes}")
        
        # Action analysis
        all_actions = np.concatenate(episode_actions, axis=0)
        action_means = np.mean(all_actions, axis=0)
        action_stds = np.std(all_actions, axis=0)
        print(f"\nAction Statistics:")
        print(f"Heating Setpoint:   {action_means[0]:8.2f} ± {action_stds[0]:6.2f}")
        print(f"Cooling Setpoint:   {action_means[1]:8.2f} ± {action_stds[1]:6.2f}")
        
        # Save results if requested
        if save_results:
            results_dir = Path(f"./evaluation_results/{Path(model_path).stem}")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            results = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'min_reward': min_reward,
                'max_reward': max_reward,
                'avg_length': avg_length,
                'std_length': std_length,
                'action_means': action_means,
                'action_stds': action_stds
            }
            
            np.savez(results_dir / "evaluation_metrics.npz", **results)
            
            # Save detailed data
            detailed_results = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_actions': episode_actions,
                'episode_observations': episode_observations
            }
            np.savez(results_dir / "detailed_results.npz", **detailed_results)
            
            # Create summary CSV
            summary_df = pd.DataFrame({
                'episode': range(1, num_episodes + 1),
                'reward': episode_rewards,
                'length': episode_lengths
            })
            summary_df.to_csv(results_dir / "episode_summary.csv", index=False)
            
            print(f"\n💾 Results saved to: {results_dir}")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': avg_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, save_path=None):
    """Plot evaluation results"""
    if results is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Reward distribution
    axes[0, 0].hist(results['episode_rewards'], bins=10, alpha=0.7, color='blue')
    axes[0, 0].axvline(results['avg_reward'], color='red', linestyle='--', label=f'Mean: {results["avg_reward"]:.2f}')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Episode length distribution
    axes[0, 1].hist(results['episode_lengths'], bins=10, alpha=0.7, color='green')
    axes[0, 1].axvline(results['avg_length'], color='red', linestyle='--', label=f'Mean: {results["avg_length"]:.2f}')
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].set_xlabel('Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Reward over episodes
    axes[1, 0].plot(results['episode_rewards'], 'b-', alpha=0.7)
    axes[1, 0].axhline(results['avg_reward'], color='red', linestyle='--', label=f'Mean: {results["avg_reward"]:.2f}')
    axes[1, 0].set_title('Reward per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    
    # Length over episodes
    axes[1, 1].plot(results['episode_lengths'], 'g-', alpha=0.7)
    axes[1, 1].axhline(results['avg_length'], color='red', linestyle='--', label=f'Mean: {results["avg_length"]:.2f}')
    axes[1, 1].set_title('Length per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Length')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained Sinergym DistilBERT PPO model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model (without .zip extension)')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes to evaluate')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results and plots')
    parser.add_argument('--plot', action='store_true',
                       help='Show/save evaluation plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 Sinergym DistilBERT PPO Model Evaluation")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Evaluate model
    results = evaluate_model_detailed(
        args.model_path, 
        num_episodes=args.num_episodes,
        save_results=args.save_results
    )
    
    # Plot results if requested
    if args.plot and results is not None:
        plot_path = None
        if args.save_results:
            plot_path = f"./evaluation_results/{Path(args.model_path).stem}/evaluation_plots.png"
        plot_results(results, save_path=plot_path)
    
    print("\n🎉 Evaluation completed!")

if __name__ == "__main__":
    main()