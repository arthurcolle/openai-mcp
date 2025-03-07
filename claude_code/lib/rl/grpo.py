"""
Group Relative Policy Optimization (GRPO) for multi-agent learning in Claude Code.
This module provides a multi-agent GRPO implementation that learns from interactions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from collections import deque
import random
import time


@dataclass
class Experience:
    """A single step of experience for reinforcement learning."""
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: Optional[Dict[str, Any]] = None


class ExperienceBuffer:
    """Buffer to store and sample experiences for training."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the experience buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    """Neural network to represent a policy."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of the input state
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the action space
        """
        super(PolicyNetwork, self).__init__()
        
        # Create the input layer
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        
        # Create hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        
        # Create output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ValueNetwork(nn.Module):
    """Neural network to represent a value function."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initialize the value network.
        
        Args:
            input_dim: Dimension of the input state
            hidden_dims: List of hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()
        
        # Create the input layer
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        
        # Create hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        
        # Create output layer (scalar value)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class GRPO:
    """
    Group Relative Policy Optimization implementation for multi-agent learning.
    GRPO extends PPO by considering relative performance within a group of agents.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_gae: bool = True,
        normalize_advantages: bool = True,
        relative_advantage_weight: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the GRPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers in networks
            lr_policy: Learning rate for policy network
            lr_value: Learning rate for value network
            gamma: Discount factor
            gae_lambda: Lambda for GAE
            clip_ratio: PPO clipping parameter
            target_kl: Target KL divergence for early stopping
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            use_gae: Whether to use GAE
            normalize_advantages: Whether to normalize advantages
            relative_advantage_weight: Weight for relative advantage component
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.normalize_advantages = normalize_advantages
        self.relative_advantage_weight = relative_advantage_weight
        self.device = device
        
        # Initialize networks
        self.policy = PolicyNetwork(state_dim, hidden_dims, action_dim).to(device)
        self.value = ValueNetwork(state_dim, hidden_dims).to(device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)
        
        # Initialize experience buffer
        self.buffer = ExperienceBuffer()
        
        # Group-level buffers for relative advantage computation
        self.group_rewards = []
        self.agent_id = None  # Will be set when joining a group
    
    def set_agent_id(self, agent_id: str) -> None:
        """Set the agent's ID within the group."""
        self.agent_id = agent_id
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Get an action from the policy for the given state.
        
        Args:
            state: The current state
            deterministic: Whether to return the most likely action
            
        Returns:
            Tuple of (action, log probability)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action distributions
        with torch.no_grad():
            logits = self.policy(state_tensor)
            distribution = Categorical(logits=logits)
            
            if deterministic:
                action = torch.argmax(logits, dim=1).item()
            else:
                action = distribution.sample().item()
                
            log_prob = distribution.log_prob(torch.tensor(action)).item()
        
        return action, log_prob
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get the estimated value of a state.
        
        Args:
            state: The state to evaluate
            
        Returns:
            The estimated value
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get value estimate
        with torch.no_grad():
            value = self.value(state_tensor).item()
        
        return value
    
    def learn(
        self,
        batch_size: int = 64,
        epochs: int = 10,
        group_rewards: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Update policy and value networks based on collected experience.
        
        Args:
            batch_size: Size of batches to use for updates
            epochs: Number of epochs to train for
            group_rewards: Rewards collected by all agents in the group
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < batch_size:
            return {"policy_loss": 0, "value_loss": 0, "kl": 0}
        
        # Prepare data for training
        states, actions, old_log_probs, returns, advantages = self._prepare_training_data(
            group_rewards)
        
        # Training metrics
        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "kl": 0,
        }
        
        # Run training for multiple epochs
        for epoch in range(epochs):
            # Generate random indices for batching
            indices = np.random.permutation(len(states))
            
            # Process in batches
            for start_idx in range(0, len(states), batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Extract batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Update policy
                policy_loss, entropy, kl = self._update_policy(
                    batch_states, batch_actions, batch_old_log_probs, batch_advantages)
                
                # Early stopping based on KL divergence
                if kl > 1.5 * self.target_kl:
                    break
                
                # Update value function
                value_loss = self._update_value(batch_states, batch_returns)
                
                # Update metrics
                metrics["policy_loss"] += policy_loss
                metrics["value_loss"] += value_loss
                metrics["entropy"] += entropy
                metrics["kl"] += kl
            
            # Check for early stopping after each epoch
            if metrics["kl"] / (epoch + 1) > self.target_kl:
                break
        
        # Normalize metrics by number of updates
        num_updates = epochs * ((len(states) + batch_size - 1) // batch_size)
        for key in metrics:
            metrics[key] /= num_updates
        
        # Clear buffer after training
        self.buffer = ExperienceBuffer()
        
        return metrics
    
    def _prepare_training_data(
        self, group_rewards: Optional[Dict[str, List[float]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training from the experience buffer.
        
        Args:
            group_rewards: Rewards collected by all agents in the group
            
        Returns:
            Tuple of (states, actions, old_log_probs, returns, advantages)
        """
        # Collect experiences from buffer
        experiences = list(self.buffer.buffer)
        
        # Extract components
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([float(exp.done) for exp in experiences]).to(self.device)
        
        # Compute values for all states and next states
        with torch.no_grad():
            values = self.value(states).squeeze()
            next_values = self.value(next_states).squeeze()
        
        # Compute advantages and returns
        if self.use_gae:
            # Generalized Advantage Estimation
            advantages = self._compute_gae(rewards, values, next_values, dones)
        else:
            # Regular advantages
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Compute returns (for value function)
        returns = advantages + values
        
        # If group rewards are provided, compute relative advantages
        if group_rewards is not None and self.agent_id in group_rewards:
            relative_advantages = self._compute_relative_advantages(
                advantages, group_rewards)
            
            # Combine regular and relative advantages
            advantages = (1 - self.relative_advantage_weight) * advantages + \
                         self.relative_advantage_weight * relative_advantages
        
        # Normalize advantages if enabled
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old log probabilities
        old_log_probs = torch.FloatTensor(
            [self._compute_log_prob(exp.state, exp.action) for exp in experiences]
        ).to(self.device)
        
        return states, actions, old_log_probs, returns, advantages
    
    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, 
        next_values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages using Generalized Advantage Estimation.
        
        Args:
            rewards: Batch of rewards
            values: Batch of state values
            next_values: Batch of next state values
            dones: Batch of done flags
            
        Returns:
            Batch of advantage estimates
        """
        # Initialize advantages
        advantages = torch.zeros_like(rewards)
        
        # Initialize gae
        gae = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            # Compute TD error
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            
            # Update gae
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            # Store advantage
            advantages[t] = gae
        
        return advantages
    
    def _compute_relative_advantages(
        self, advantages: torch.Tensor, group_rewards: Dict[str, List[float]]
    ) -> torch.Tensor:
        """
        Compute relative advantages compared to other agents in the group.
        
        Args:
            advantages: This agent's advantages
            group_rewards: Rewards collected by all agents in the group
            
        Returns:
            Relative advantages
        """
        # Compute mean reward for each agent
        agent_mean_rewards = {
            agent_id: sum(rewards) / max(1, len(rewards))
            for agent_id, rewards in group_rewards.items()
        }
        
        # Compute mean reward across all agents
        group_mean_reward = sum(agent_mean_rewards.values()) / len(agent_mean_rewards)
        
        # Compute relative performance factor
        # Higher if this agent is doing better than the group average
        if self.agent_id in agent_mean_rewards:
            relative_factor = agent_mean_rewards[self.agent_id] / (group_mean_reward + 1e-8)
        else:
            relative_factor = 1.0
        
        # Apply the relative factor to the advantages
        relative_advantages = advantages * relative_factor
        
        return relative_advantages
    
    def _compute_log_prob(self, state: np.ndarray, action: int) -> float:
        """
        Compute the log probability of an action given a state.
        
        Args:
            state: The state
            action: The action
            
        Returns:
            The log probability
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action distribution
        with torch.no_grad():
            logits = self.policy(state_tensor)
            distribution = Categorical(logits=logits)
            log_prob = distribution.log_prob(torch.tensor(action, device=self.device)).item()
        
        return log_prob
    
    def _update_policy(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        old_log_probs: torch.Tensor, 
        advantages: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Update the policy network using PPO.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old log probabilities
            advantages: Batch of advantages
            
        Returns:
            Tuple of (policy_loss, entropy, kl_divergence)
        """
        # Get action distributions
        logits = self.policy(states)
        distribution = Categorical(logits=logits)
        
        # Get new log probabilities
        new_log_probs = distribution.log_prob(actions)
        
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute surrogate objectives
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # Compute policy loss (negative because we're maximizing)
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Compute entropy bonus
        entropy = distribution.entropy().mean()
        
        # Add entropy bonus to loss
        loss = policy_loss - self.entropy_coef * entropy
        
        # Compute approximate KL divergence for monitoring
        with torch.no_grad():
            kl = (old_log_probs - new_log_probs).mean().item()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        return policy_loss.item(), entropy.item(), kl
    
    def _update_value(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """
        Update the value network.
        
        Args:
            states: Batch of states
            returns: Batch of returns
            
        Returns:
            Value loss
        """
        # Get value predictions
        values = self.value(states).squeeze()
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns)
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        return value_loss.item()


class MultiAgentGroupRL:
    """
    Multi-agent reinforcement learning system using GRPO for Claude Code.
    This class manages multiple GRPO agents that learn in a coordinated way.
    """
    
    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        feature_extractor: Callable[[Dict[str, Any]], np.ndarray],
        reward_function: Callable[[Dict[str, Any], str, Any], float],
        update_interval: int = 1000,
        training_epochs: int = 10,
        batch_size: int = 64,
        save_dir: str = "./models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the multi-agent RL system.
        
        Args:
            agent_configs: List of configurations for each agent
            feature_extractor: Function to extract state features
            reward_function: Function to compute rewards
            update_interval: How often to update agents (in steps)
            training_epochs: Number of epochs to train for each update
            batch_size: Batch size for training
            save_dir: Directory to save models
            device: Device to run on
        """
        self.feature_extractor = feature_extractor
        self.reward_function = reward_function
        self.update_interval = update_interval
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.device = device
        
        # Initialize agents
        self.agents = {}
        for config in agent_configs:
            agent_id = config["id"]
            state_dim = config["state_dim"]
            action_dim = config["action_dim"]
            
            # Create GRPO agent
            agent = GRPO(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.get("hidden_dims", [64, 64]),
                device=device,
                **{k: v for k, v in config.items() if k not in ["id", "state_dim", "action_dim", "hidden_dims"]}
            )
            
            # Set agent ID
            agent.set_agent_id(agent_id)
            
            self.agents[agent_id] = agent
        
        # Track steps for periodic updates
        self.total_steps = 0
        
        # Store rewards for relative advantage computation
        self.agent_rewards = {agent_id: [] for agent_id in self.agents}
    
    def select_action(
        self, agent_id: str, observation: Dict[str, Any], deterministic: bool = False
    ) -> Tuple[Any, float]:
        """
        Select an action for the specified agent.
        
        Args:
            agent_id: ID of the agent
            observation: Current observation
            deterministic: Whether to select deterministically
            
        Returns:
            Tuple of (action, log probability)
        """
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        # Extract features
        state = self.feature_extractor(observation)
        
        # Get action from agent
        action, log_prob = self.agents[agent_id].get_action(state, deterministic)
        
        return action, log_prob
    
    def observe(
        self, 
        agent_id: str, 
        observation: Dict[str, Any],
        action: Any,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an observation for the specified agent.
        
        Args:
            agent_id: ID of the agent
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            info: Additional information
        """
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        # Extract features
        state = self.feature_extractor(observation)
        next_state = self.feature_extractor(next_observation)
        
        # Create experience
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info
        )
        
        # Add experience to agent's buffer
        self.agents[agent_id].buffer.add(exp)
        
        # Store reward for relative advantage computation
        self.agent_rewards[agent_id].append(reward)
        
        # Increment step counter
        self.total_steps += 1
        
        # Perform updates if needed
        if self.total_steps % self.update_interval == 0:
            self.update_all_agents()
    
    def update_all_agents(self) -> Dict[str, Dict[str, float]]:
        """
        Update all agents' policies.
        
        Returns:
            Dictionary of training metrics for each agent
        """
        # Store metrics for each agent
        metrics = {}
        
        # Update each agent
        for agent_id, agent in self.agents.items():
            # Train the agent with group rewards
            agent_metrics = agent.learn(
                batch_size=self.batch_size,
                epochs=self.training_epochs,
                group_rewards=self.agent_rewards
            )
            
            metrics[agent_id] = agent_metrics
        
        # Reset reward tracking
        self.agent_rewards = {agent_id: [] for agent_id in self.agents}
        
        return metrics
    
    def save_agents(self, suffix: str = "") -> None:
        """
        Save all agents' models.
        
        Args:
            suffix: Optional suffix for saved files
        """
        import os
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save each agent
        for agent_id, agent in self.agents.items():
            # Create file path
            file_path = os.path.join(self.save_dir, f"{agent_id}{suffix}.pt")
            
            # Save model
            torch.save({
                "policy_state_dict": agent.policy.state_dict(),
                "value_state_dict": agent.value.state_dict(),
                "policy_optimizer_state_dict": agent.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": agent.value_optimizer.state_dict(),
            }, file_path)
    
    def load_agents(self, suffix: str = "") -> None:
        """
        Load all agents' models.
        
        Args:
            suffix: Optional suffix for loaded files
        """
        import os
        
        # Load each agent
        for agent_id, agent in self.agents.items():
            # Create file path
            file_path = os.path.join(self.save_dir, f"{agent_id}{suffix}.pt")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: Model file not found for agent {agent_id}")
                continue
            
            # Load model
            checkpoint = torch.load(file_path, map_location=self.device)
            
            # Load state dicts
            agent.policy.load_state_dict(checkpoint["policy_state_dict"])
            agent.value.load_state_dict(checkpoint["value_state_dict"])
            agent.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
            agent.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])


class ToolSelectionGRPO:
    """
    Specialized GRPO implementation for tool selection in Claude Code.
    This class adapts the MultiAgentGroupRL for the specific context of tool selection.
    """
    
    def __init__(
        self,
        tool_registry: Any,  # Should be a reference to the tool registry
        context_evaluator: Callable,  # Function to evaluate quality of response given context
        state_dim: int = 768,  # Embedding dimension for query
        num_agents: int = 3,  # Number of agents in the group
        update_interval: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the GRPO tool selector.
        
        Args:
            tool_registry: Registry containing available tools
            context_evaluator: Function to evaluate response quality
            state_dim: Dimension of state features
            num_agents: Number of agents in the group
            update_interval: How often to update agents
            device: Device to run on
        """
        self.tool_registry = tool_registry
        self.context_evaluator = context_evaluator
        
        # Get all available tools
        self.tool_names = tool_registry.get_all_tool_names()
        self.action_dim = len(self.tool_names)
        
        # Define agent configurations
        agent_configs = [
            {
                "id": f"tool_agent_{i}",
                "state_dim": state_dim,
                "action_dim": self.action_dim,
                "hidden_dims": [256, 128],
                "relative_advantage_weight": 0.7 if i > 0 else 0.3,  # Different weights
                "entropy_coef": 0.02 if i == 0 else 0.01,  # Different exploration rates
            }
            for i in range(num_agents)
        ]
        
        # Initialize multi-agent RL system
        self.rl_system = MultiAgentGroupRL(
            agent_configs=agent_configs,
            feature_extractor=self._extract_features,
            reward_function=self._compute_reward,
            update_interval=update_interval,
            device=device,
        )
        
        # Track current episode
        self.current_episode = {agent_id: {} for agent_id in self.rl_system.agents}
    
    def select_tool(self, user_query: str, context: Dict[str, Any], visualizer=None) -> str:
        """
        Select the best tool to use for a given user query and context.
        
        Args:
            user_query: The user's query
            context: The current conversation context
            visualizer: Optional visualizer to display the selection process
            
        Returns:
            The name of the best tool to use
        """
        # Create observation
        observation = {
            "query": user_query,
            "context": context,
        }
        
        # If visualizer is provided, start it
        if visualizer:
            visualizer.start()
            visualizer.add_execution(
                execution_id="tool_selection",
                tool_name="GRPO Tool Selection",
                parameters={"query": user_query[:100] + "..." if len(user_query) > 100 else user_query}
            )
        
        # Select agent to use (round-robin for now)
        agent_id = f"tool_agent_{self.rl_system.total_steps % len(self.rl_system.agents)}"
        
        # Update visualizer if provided
        if visualizer:
            visualizer.update_progress("tool_selection", 0.3)
        
        # Get action from agent
        action_idx, _ = self.rl_system.select_action(
            agent_id=agent_id,
            observation=observation,
            deterministic=False  # Use exploratory actions during learning
        )
        
        # Update visualizer if provided
        if visualizer:
            visualizer.update_progress("tool_selection", 0.6)
        
        # Store initial information for the episode
        self.current_episode[agent_id] = {
            "observation": observation,
            "action_idx": action_idx,
            "initial_quality": self.context_evaluator(context),
        }
        
        # Map action index to tool name
        tool_name = self.tool_names[action_idx]
        
        # Complete visualization if provided
        if visualizer:
            # Create detailed metrics for visualization
            agent_data = {}
            for aid, agent in self.rl_system.agents.items():
                # Get all tool probabilities for this agent
                with torch.no_grad():
                    state = self.rl_system._extract_features(observation)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    logits = agent.policy(state_tensor)
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                # Add to metrics
                agent_data[aid] = {
                    "selected": aid == agent_id,
                    "tool_probabilities": {
                        self.tool_names[i]: float(prob) 
                        for i, prob in enumerate(probs)
                    }
                }
            
            # Complete the visualization
            visualizer.complete_execution(
                execution_id="tool_selection",
                result={
                    "selected_tool": tool_name,
                    "selected_agent": agent_id,
                    "agent_data": agent_data
                },
                status="success"
            )
            visualizer.stop()
        
        return tool_name
    
    def observe_result(
        self, agent_id: str, result: Any, context: Dict[str, Any], done: bool = True
    ) -> None:
        """
        Observe the result of using a tool.
        
        Args:
            agent_id: The ID of the agent that selected the tool
            result: The result of using the tool
            context: The updated context after using the tool
            done: Whether the interaction is complete
        """
        if agent_id not in self.current_episode:
            return
        
        # Get episode information
        episode = self.current_episode[agent_id]
        observation = episode["observation"]
        action_idx = episode["action_idx"]
        initial_quality = episode["initial_quality"]
        
        # Create next observation
        next_observation = {
            "query": observation["query"],
            "context": context,
            "result": result,
        }
        
        # Compute reward
        reward = self._compute_reward(observation, action_idx, result, context, initial_quality)
        
        # Record observation
        self.rl_system.observe(
            agent_id=agent_id,
            observation=observation,
            action=action_idx,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )
        
        # Clear episode if done
        if done:
            self.current_episode[agent_id] = {}
    
    def _extract_features(self, observation: Dict[str, Any]) -> np.ndarray:
        """Extract features from an observation."""
        # This would ideally use an embedding model
        # For now, return a random vector as a placeholder
        return np.random.randn(768)
    
    def _compute_reward(
        self, 
        observation: Dict[str, Any], 
        action_idx: int, 
        result: Any,
        context: Dict[str, Any], 
        initial_quality: float
    ) -> float:
        """Compute the reward for an action."""
        # Compute the quality improvement
        final_quality = self.context_evaluator(context)
        quality_improvement = final_quality - initial_quality
        
        # Base reward on quality improvement
        reward = max(0, quality_improvement * 10)  # Scale for better learning
        
        return reward
    
    def update(self) -> Dict[str, Dict[str, float]]:
        """
        Trigger an update of all agents.
        
        Returns:
            Dictionary of training metrics
        """
        return self.rl_system.update_all_agents()
    
    def save(self, suffix: str = "") -> None:
        """Save all agents."""
        self.rl_system.save_agents(suffix)
    
    def load(self, suffix: str = "") -> None:
        """Load all agents."""
        self.rl_system.load_agents(suffix)