import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from learning_env import SpeakerEnvironment
from utils.audio_utils import extract_segments, embed_audio

class Actor(nn.Module):
    """Actor network for the RL agent. Outputs the action probabilities."""
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(Actor, self).__init__()

        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, state):
        """Forward pass through the network."""
        # Input validation - replace NaN values
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        # Ensure valid probabilities
        action_probs = torch.softmax(x, dim=1)
        
        # Additional safety check
        if torch.isnan(action_probs).any():
            # If NaN detected, use uniform distribution
            action_probs = torch.ones_like(action_probs) / action_probs.shape[1]
        
        return action_probs

    def select_action(self, state, training = True):
        """Select an action based on the state."""
        # Ensure state has correct shape (batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        action_probs = self.forward(state)

        if training:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action.item(), log_prob
        else:
            action = torch.argmax(action_probs, dim=1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)))
            return action.item(), log_prob

class Critic(nn.Module):
    """
    Critic Network (Value Network)
    Takes state as input and outputs value estimate
    """
    def __init__(self, state_size, hidden_size=128):
        super(Critic, self).__init__()
        
        # Define the neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output single value
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        """
        Forward pass through the critic network
        Args:
            state: Current state
        Returns:
            value: Estimated value of the state
        """
        # First layer with ReLU activation
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        
        # Second layer with ReLU activation
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer - no activation (value can be negative)
        value = self.fc3(x)
        
        return value   

class ActorCriticAgent:
    """
    Actor-Critic Agent for Speaker Diarization
    Combines policy learning (Actor) with value estimation (Critic)
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Initialize networks
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        
        # Optimizers for each network
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer for storing transitions
        self.memory = []
        
    def select_action(self, state, training=True):
        """
        Select action using the actor network
        Args:
            state: Current state
            training: Whether in training mode
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate of current state
        """
        # Convert state to tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        # Get action from actor
        action, log_prob = self.actor.select_action(state, training)
        
        # Get value estimate from critic
        value = self.critic(state)
        
        return action, log_prob, value
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in memory
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert state to numpy array if it's a tensor
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def update(self, gamma=0.99):
        """
        Update both actor and critic networks
        Args:
            gamma: Discount factor for future rewards
        """
        if len(self.memory) < 1:  # Need at least 1 transition for faster learning
            return
        
        # Filter out transitions where next_state is None
        valid_transitions = [t for t in self.memory if t['next_state'] is not None]
        
        if len(valid_transitions) < 1:  # Allow single transition updates
            return
        
        # Convert memory to tensors (convert to numpy arrays first for efficiency)
        states = torch.FloatTensor(np.array([t['state'] for t in valid_transitions])).to(self.device)
        actions = torch.LongTensor([t['action'] for t in valid_transitions]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in valid_transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t['next_state'] for t in valid_transitions])).to(self.device)
        dones = torch.BoolTensor([t['done'] for t in valid_transitions]).to(self.device)
        
        # Calculate returns (discounted rewards)
        returns = []
        cumulative_return = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                cumulative_return = 0
            cumulative_return = reward + gamma * cumulative_return
            returns.insert(0, cumulative_return)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get current values and action probabilities
        values = self.critic(states).squeeze()
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # Calculate advantages (TD error)
        advantages = returns - values.detach()
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns)
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Clear memory after update
        self.memory = []
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filepath):
        """Save the agent's networks"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load the agent's networks"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
       
