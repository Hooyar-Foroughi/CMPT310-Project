import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path 
sys.path.append('src')

from learning_env import SpeakerEnvironment
from rl_agent import ActorCriticAgent
from rl_trainer import RLTrainer

def main():
    """Main training script"""
    print("RL Speaker Diarization Training")
    
    
    # Configuration
    audio_file = "audio/cjfer.wav"
    segment_duration = 0.5  # Slightly longer for Resemblyzer
    num_episodes = 15  # Fewer episodes for testing
    learning_rate = 0.001  # More stable learning rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f" Configuration:")
    print(f"Audio file: {audio_file}")
    print(f"Segment duration: {segment_duration}s")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found!")
        return
    
    # Create environment
    print(f"\nCreating environment...")
    env = SpeakerEnvironment(audio_file, segment_duration=segment_duration)
    print(f"Environment created with {env.n_segments} segments")
    
    # Determine state and action sizes
    test_state = env.reset()
    if test_state is None:
        print("Failed to get initial state!")
        return
    
    state_size = test_state.shape[0]
    action_size = 5  # 5 possible speakers (0-4)
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent
    print(f"\nCreating RL agent...")
    agent = ActorCriticAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        device=device
    )
    print(f"Agent created successfully")
    
    # Create trainer
    print(f"\n Creating trainer...")
    trainer = RLTrainer(agent, env, device=device)
    print(f"Trainer created successfully")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Train the agent
    print(f"\nStarting training...")
    episode_rewards, actor_losses, critic_losses = trainer.train(
        num_episodes=num_episodes,
        save_interval=5,  # Save more frequently
        plot_interval=5    # Plot more frequently
    )
    
    # Evaluate the trained agent
    print(f"\nEvaluating trained agent...")
    avg_reward, results = trainer.evaluate(num_episodes=5)
    
    # Save final model
    agent.save('models/final_rl_agent.pth')
    print(f"Final model saved!")
    
    # Print some results
    print(f"\nSample Results:")
    for i, result in enumerate(results[:3]):  # Show first 3 episodes
        print(f"   Episode {i+1}: {len(result)} segments, {len(set(result))} unique speakers")
        print(f"   Speaker distribution: {dict(zip(*np.unique(result, return_counts=True)))}")
    
    print(f"\nTraining complete! Check models/ and visualizations/ folders for results.")

if __name__ == "__main__":
    main()





