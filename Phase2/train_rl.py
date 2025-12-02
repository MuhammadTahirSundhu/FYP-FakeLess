"""
===================================================================================
MODELS: RL AGENT
===================================================================================
File: models/rl_agent.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class RLPolicyNetwork(nn.Module):
    """Actor network for perturbation parameter selection"""
    def __init__(self, state_dim=256, action_dim=64, hidden_dims=[512, 512, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output mean and std for continuous actions
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        
    def forward(self, state):
        x = self.encoder(state)
        mean = torch.tanh(self.mean_layer(x))  # Bounded actions
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.exp(log_std)
        return mean, std
    
    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate_action(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class RLValueNetwork(nn.Module):
    """Critic network for value estimation"""
    def __init__(self, state_dim=256, hidden_dims=[512, 512, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state).squeeze(-1)


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    def __init__(self, state_dim=256, action_dim=64, device='cuda',
                 lr_policy=3e-4, lr_value=1e-3, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2):
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        
        # Networks
        self.policy = RLPolicyNetwork(state_dim, action_dim).to(device)
        self.value = RLValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
    def select_action(self, state):
        """Select action during training"""
        with torch.no_grad():
            action, log_prob = self.policy.sample_action(state)
        return action, log_prob
    
    def evaluate(self, state):
        """Get value estimate"""
        with torch.no_grad():
            value = self.value(state)
        return value
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        lastgae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[-1]
                nextvalue = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalue = values[t+1]
            
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            advantages[t] = lastgae = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns, 
               epochs=10, batch_size=64):
        """PPO update"""
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Policy loss
                log_probs, entropy = self.policy.evaluate_action(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy_loss = -0.01 * entropy.mean()
                
                # Value loss
                values = self.value(batch_states)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Update policy
                self.policy_optimizer.zero_grad()
                (policy_loss + entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.value_optimizer.step()