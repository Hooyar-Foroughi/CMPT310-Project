import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class RLTrainer:
    """
    Training loop for the RL Speaker Diarization Agent
    """
    def __init__(self, agent, env, device='cpu'):
        self.agent = agent
        self.env = env
        self.device = device
        
        # Training metrics
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.episode_lengths = []
        
    def train_episode(self):
        """
        Train for one episode
        Returns:
            episode_reward: Total reward for the episode
            episode_length: Number of steps in episode
        """
        # Reset environment
        state = self.env.reset()
        if state is None:
            return 0, 0
            
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            action, log_prob, value = self.agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Move to next state
            state = next_state
            
            # Update networks more frequently for faster learning
            if episode_length % 5 == 0:  # Update every 5 steps
                actor_loss, critic_loss = self.agent.update()
                if actor_loss is not None:
                    self.actor_losses.append(actor_loss)
                    self.critic_losses.append(critic_loss)
            
            # Check if episode is done
            if done or next_state is None:
                break
        
        return episode_reward, episode_length
    
    def train(self, num_episodes=1000, save_interval=100, plot_interval=50):
        """
        Main training loop
        Args:
            num_episodes: Number of episodes to train
            save_interval: How often to save the model
            plot_interval: How often to plot training curves
        """
        print(f"Starting RL Training for {num_episodes} episodes")
        print(f"State size: {self.agent.state_size}, Action size: {self.agent.action_size}")
        
        best_reward = float('-inf')
        
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            # Train one episode
            episode_reward, episode_length = self.train_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress more frequently
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(self.episode_rewards[-5:])
                avg_length = np.mean(self.episode_lengths[-5:])
                print(f"\nEpisode {episode + 1}: Avg Reward = {avg_reward:.3f}, Avg Length = {avg_length:.1f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save('models/best_rl_agent.pth')
                print(f"New best model saved! Reward: {best_reward:.3f}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self.agent.save(f'models/rl_agent_episode_{episode + 1}.pth')
            
            # Plot training curves
            if (episode + 1) % plot_interval == 0:
                self.plot_training_curves()
        
        print(f"\nTraining completed!")
        print(f"Best episode reward: {best_reward:.3f}")
        print(f"Average episode reward: {np.mean(self.episode_rewards):.3f}")
        
        return self.episode_rewards, self.actor_losses, self.critic_losses
    
    def plot_training_curves(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Moving average of rewards
        if len(self.episode_rewards) > 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Moving Average Rewards (window=10)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Loss curves
        if len(self.actor_losses) > 0:
            axes[1, 1].plot(self.actor_losses, label='Actor Loss')
            axes[1, 1].plot(self.critic_losses, label='Critic Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/rl_training_curves.png')
        plt.close()
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained agent
        Args:
            num_episodes: Number of episodes to evaluate
        Returns:
            avg_reward: Average reward across episodes
            results: List of speaker assignments
        """
        print(f"Evaluating agent over {num_episodes} episodes...")
        
        eval_rewards = []
        all_results = []
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            if state is None:
                continue
                
            episode_reward = 0
            episode_results = []
            
            while True:
                # Select action (no exploration during evaluation)
                action, _, _ = self.agent.select_action(state, training=False)
                
                # Take step
                next_state, reward, done = self.env.step(action)
                
                # Store results
                episode_results.append(action)
                episode_reward += reward
                
                # Move to next state
                state = next_state
                
                if done or next_state is None:
                    break
            
            eval_rewards.append(episode_reward)
            all_results.append(episode_results)
        
        avg_reward = np.mean(eval_rewards)
        print(f"Evaluation Results:")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Reward Std: {np.std(eval_rewards):.3f}")
        
        return avg_reward, all_results