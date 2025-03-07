"""
Monte Carlo Tree Search implementation for decision making in Claude Code.
This module provides an advanced MCTS implementation that can be used to select
optimal actions/tools based on simulated outcomes.
"""

import math
import numpy as np
import random
from typing import List, Dict, Any, Callable, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class MCTSNode:
    """Represents a node in the Monte Carlo search tree."""
    state: Any
    parent: Optional['MCTSNode'] = None
    action_taken: Any = None
    visits: int = 0
    value: float = 0.0
    children: Dict[Any, 'MCTSNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    def is_fully_expanded(self, possible_actions: List[Any]) -> bool:
        """Check if all possible actions have been tried from this node."""
        return all(action in self.children for action in possible_actions)
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        # This should be customized based on your environment
        return False
    
    def best_child(self, exploration_weight: float = 1.0) -> 'MCTSNode':
        """Select the best child node according to UCB1 formula."""
        if not self.children:
            return None
            
        def ucb_score(child: MCTSNode) -> float:
            exploitation = child.value / child.visits if child.visits > 0 else 0
            exploration = math.sqrt(2 * math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            return exploitation + exploration_weight * exploration
            
        return max(self.children.values(), key=ucb_score)


class AdvancedMCTS:
    """
    Advanced Monte Carlo Tree Search implementation with various enhancements:
    - Progressive widening for large/continuous action spaces
    - RAVE (Rapid Action Value Estimation)
    - Parallel simulations
    - Dynamic exploration weight
    - Customizable simulation and backpropagation strategies
    """
    
    def __init__(
        self, 
        state_evaluator: Callable[[Any], float],
        action_generator: Callable[[Any], List[Any]],
        simulator: Callable[[Any, Any], Any],
        max_iterations: int = 1000,
        exploration_weight: float = 1.0,
        time_limit: Optional[float] = None,
        progressive_widening: bool = False,
        pw_coef: float = 0.5,
        pw_power: float = 0.5,
        use_rave: bool = False,
        rave_equiv_param: float = 1000,
    ):
        """
        Initialize the MCTS algorithm.
        
        Args:
            state_evaluator: Function to evaluate the value of a state (terminal or not)
            action_generator: Function to generate possible actions from a state
            simulator: Function to simulate taking an action in a state, returning new state
            max_iterations: Maximum number of search iterations
            exploration_weight: Controls exploration vs exploitation balance
            time_limit: Optional time limit for search in seconds
            progressive_widening: Whether to use progressive widening for large action spaces
            pw_coef: Coefficient for progressive widening
            pw_power: Power for progressive widening
            use_rave: Whether to use RAVE (Rapid Action Value Estimation)
            rave_equiv_param: RAVE equivalence parameter
        """
        self.state_evaluator = state_evaluator
        self.action_generator = action_generator 
        self.simulator = simulator
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        
        # Progressive widening parameters
        self.progressive_widening = progressive_widening
        self.pw_coef = pw_coef
        self.pw_power = pw_power
        
        # RAVE parameters
        self.use_rave = use_rave
        self.rave_equiv_param = rave_equiv_param
        self.rave_values = {}  # (state, action) -> (value, visits)
    
    def search(self, initial_state: Any, visualizer=None) -> Any:
        """
        Perform MCTS search from the initial state and return the best action.
        
        Args:
            initial_state: The starting state for the search
            visualizer: Optional visualizer to show progress
            
        Returns:
            The best action found by the search
        """
        root = MCTSNode(state=initial_state)
        
        # Initialize visualizer if provided
        if visualizer:
            visualizer.set_search_parameters(root, self.max_iterations)
        
        # Run iterations of the MCTS algorithm
        for iteration in range(self.max_iterations):
            # Selection phase
            selected_node = self._select(root)
            
            # Expansion phase (if not terminal)
            expanded_node = None
            if not selected_node.is_terminal():
                expanded_node = self._expand(selected_node)
            else:
                expanded_node = selected_node
            
            # Simulation phase
            simulation_path = []
            if visualizer:
                # Track simulation path for visualization
                current = expanded_node
                current_state = current.state
                while current.parent:
                    simulation_path.insert(0, (current.parent.state, current.action_taken))
                    current = current.parent
            
            simulation_result = self._simulate(expanded_node)
            
            # Backpropagation phase
            self._backpropagate(expanded_node, simulation_result)
            
            # Update visualization
            if visualizer:
                # Find current best action
                best_action = None
                if root.children:
                    best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
                
                # Update visualizer
                visualizer.update_iteration(
                    iteration=iteration + 1,
                    selected_node=selected_node,
                    expanded_node=expanded_node,
                    simulation_path=simulation_path,
                    simulation_result=simulation_result,
                    best_action=best_action
                )
            
        # Return the action that leads to the child with the highest value
        if not root.children:
            possible_actions = self.action_generator(root.state)
            if possible_actions:
                best_action = random.choice(possible_actions)
                if visualizer:
                    visualizer.update_iteration(
                        iteration=self.max_iterations,
                        best_action=best_action
                    )
                return best_action
            return None
        
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        if visualizer:
            visualizer.update_iteration(
                iteration=self.max_iterations,
                best_action=best_action
            )
        return best_action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to expand using UCB1 and progressive widening if enabled.
        
        Args:
            node: The current node
            
        Returns:
            The selected node for expansion
        """
        while not node.is_terminal():
            possible_actions = self.action_generator(node.state)
            
            # Handle progressive widening if enabled
            if self.progressive_widening:
                max_children = max(1, int(self.pw_coef * (node.visits ** self.pw_power)))
                if len(node.children) < min(max_children, len(possible_actions)):
                    return node
            
            # If not fully expanded, select this node for expansion
            if not node.is_fully_expanded(possible_actions):
                return node
                
            # Otherwise, select the best child according to UCB1
            node = node.best_child(self.exploration_weight)
            if node is None:
                break
                
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand the node by selecting an untried action and creating a new child node.
        
        Args:
            node: The node to expand
            
        Returns:
            The newly created child node
        """
        possible_actions = self.action_generator(node.state)
        untried_actions = [a for a in possible_actions if a not in node.children]
        
        if not untried_actions:
            return node
            
        action = random.choice(untried_actions)
        new_state = self.simulator(node.state, action)
        child_node = MCTSNode(
            state=new_state,
            parent=node,
            action_taken=action
        )
        node.children[action] = child_node
        return child_node
    
    def _simulate(self, node: MCTSNode, depth: int = 10) -> float:
        """
        Simulate a random playout from the given node until a terminal state or max depth.
        
        Args:
            node: The node to start simulation from
            depth: Maximum simulation depth
            
        Returns:
            The value of the simulated outcome
        """
        state = node.state
        current_depth = 0
        
        # Continue simulation until we reach a terminal state or max depth
        while current_depth < depth:
            if self._is_terminal_state(state):
                break
                
            possible_actions = self.action_generator(state)
            if not possible_actions:
                break
                
            action = random.choice(possible_actions)
            state = self.simulator(state, action)
            current_depth += 1
            
        return self.state_evaluator(state)
    
    def _is_terminal_state(self, state: Any) -> bool:
        """Determine if the state is terminal."""
        # This should be customized based on your environment
        return False
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate the simulation result up the tree.
        
        Args:
            node: The leaf node where simulation started
            value: The value from the simulation
        """
        while node is not None:
            node.visits += 1
            node.value += value
            
            # Update RAVE values if enabled
            if self.use_rave and node.parent is not None:
                state_hash = self._hash_state(node.parent.state)
                action = node.action_taken
                if (state_hash, action) not in self.rave_values:
                    self.rave_values[(state_hash, action)] = [0, 0]  # [value, visits]
                rave_value, rave_visits = self.rave_values[(state_hash, action)]
                self.rave_values[(state_hash, action)] = [
                    rave_value + value,
                    rave_visits + 1
                ]
                
            node = node.parent
    
    def _hash_state(self, state: Any) -> int:
        """Create a hash of the state for RAVE table lookups."""
        # This should be customized based on your state representation
        if hasattr(state, "__hash__"):
            return hash(state)
        return hash(str(state))


class MCTSToolSelector:
    """
    Specialized MCTS implementation for selecting optimal tools in Claude Code.
    This class adapts the AdvancedMCTS for the specific context of tool selection.
    """
    
    def __init__(
        self,
        tool_registry: Any,  # Should be a reference to the tool registry
        context_evaluator: Callable,  # Function to evaluate quality of response given context
        max_iterations: int = 200,
        exploration_weight: float = 1.0,
        use_learning: bool = True,
        tool_history_weight: float = 0.7,
        enable_plan_generation: bool = True,
        use_semantic_similarity: bool = True,
        adaptation_rate: float = 0.05
    ):
        """
        Initialize the MCTS tool selector with enhanced intelligence.
        
        Args:
            tool_registry: Registry containing available tools
            context_evaluator: Function to evaluate response quality
            max_iterations: Maximum search iterations
            exploration_weight: Controls exploration vs exploitation
            use_learning: Whether to use learning from past tool selections
            tool_history_weight: Weight given to historical tool performance
            enable_plan_generation: Generate complete tool sequences as plans
            use_semantic_similarity: Use semantic similarity for tool relevance
            adaptation_rate: Rate at which the system adapts to new patterns
        """
        self.tool_registry = tool_registry
        self.context_evaluator = context_evaluator
        self.use_learning = use_learning
        self.tool_history_weight = tool_history_weight
        self.enable_plan_generation = enable_plan_generation
        self.use_semantic_similarity = use_semantic_similarity
        self.adaptation_rate = adaptation_rate
        
        # Tool performance history by query type
        self.tool_history = {}
        
        # Tool sequence effectiveness records
        self.sequence_effectiveness = {}
        
        # Semantic fingerprints for tools and queries
        self.tool_fingerprints = {}
        self.query_clusters = {}
        
        # Cached simulation results for similar queries
        self.simulation_cache = {}
        
        # Initialize the MCTS algorithm
        self.mcts = AdvancedMCTS(
            state_evaluator=self._evaluate_state,
            action_generator=self._generate_actions,
            simulator=self._simulate_action,
            max_iterations=max_iterations,
            exploration_weight=exploration_weight,
            progressive_widening=True
        )
        
        # Initialize tool fingerprints
        self._initialize_tool_fingerprints()
    
    def _initialize_tool_fingerprints(self):
        """Initialize semantic fingerprints for all available tools."""
        if not self.use_semantic_similarity:
            return
            
        for tool_name in self.tool_registry.get_all_tool_names():
            tool = self.tool_registry.get_tool(tool_name)
            if tool and hasattr(tool, 'description'):
                # In a real implementation, this would compute an embedding
                # For now, we'll use a simple keyword extraction as a placeholder
                keywords = set(word.lower() for word in tool.description.split() 
                             if len(word) > 3)
                self.tool_fingerprints[tool_name] = {
                    'keywords': keywords,
                    'description': tool.description,
                    'usage_contexts': set()
                }
    
    def select_tool(self, user_query: str, context: Dict[str, Any], visualizer=None) -> Union[str, List[str]]:
        """
        Select the best tool to use for a given user query and context.
        
        Args:
            user_query: The user's query
            context: The current conversation context
            visualizer: Optional visualizer to show the selection process
            
        Returns:
            Either a single tool name or a sequence of tool names (if plan generation is enabled)
        """
        # Analyze query to determine its type/characteristics
        query_type = self._analyze_query(user_query)
        
        # Update semantic fingerprints with this query
        if self.use_semantic_similarity:
            self._update_query_clusters(user_query, query_type)
        
        initial_state = {
            'query': user_query,
            'query_type': query_type,
            'context': context,
            'actions_taken': [],
            'response_quality': 0.0,
            'steps_remaining': 3 if self.enable_plan_generation else 1,
            'step_results': {}
        }
        
        # First check if we have a high-confidence cached result for similar queries
        cached_result = self._check_cache(user_query, query_type)
        if cached_result and random.random() > 0.1:  # 10% random exploration
            if visualizer:
                visualizer.add_execution(
                    execution_id="mcts_cache_hit",
                    tool_name="MCTS Tool Selection (cached)",
                    parameters={"query": user_query[:100] + "..." if len(user_query) > 100 else user_query}
                )
                visualizer.complete_execution(
                    execution_id="mcts_cache_hit",
                    result={"selected_tool": cached_result, "source": "cache"},
                    status="success"
                )
            return cached_result
        
        # Run MCTS search
        best_action = self.mcts.search(initial_state, visualizer)
        
        # If plan generation is enabled, we might want to return a sequence
        if self.enable_plan_generation:
            # Extract the most promising action sequence from search
            plan = self._extract_plan_from_search()
            if plan and len(plan) > 1:
                # Store this plan in our cache
                self._cache_result(user_query, query_type, plan)
                return plan
        
        # Store single action in cache
        self._cache_result(user_query, query_type, best_action)
        return best_action
    
    def _analyze_query(self, query: str) -> str:
        """
        Analyze a query to determine its type and characteristics.
        
        Args:
            query: The user query
            
        Returns:
            A string identifying the query type
        """
        query_lower = query.lower()
        
        # Check for search-related queries
        if any(term in query_lower for term in ['find', 'search', 'where', 'look for']):
            return 'search'
            
        # Check for explanation queries
        if any(term in query_lower for term in ['explain', 'how', 'why', 'what is']):
            return 'explanation'
            
        # Check for file operation queries
        if any(term in query_lower for term in ['file', 'read', 'write', 'edit', 'create']):
            return 'file_operation'
            
        # Check for execution queries
        if any(term in query_lower for term in ['run', 'execute', 'start']):
            return 'execution'
            
        # Check for debugging queries
        if any(term in query_lower for term in ['debug', 'fix', 'error', 'problem']):
            return 'debugging'
            
        # Default to general
        return 'general'
    
    def _update_query_clusters(self, query: str, query_type: str):
        """
        Update query clusters with new query information.
        
        Args:
            query: The user query
            query_type: The type of query
        """
        # Extract query keywords
        keywords = set(word.lower() for word in query.split() if len(word) > 3)
        
        # Update query clusters
        if query_type not in self.query_clusters:
            self.query_clusters[query_type] = {
                'keywords': set(),
                'queries': []
            }
            
        # Add keywords to cluster
        self.query_clusters[query_type]['keywords'].update(keywords)
        
        # Add query to cluster (limit to last 50)
        self.query_clusters[query_type]['queries'].append(query)
        if len(self.query_clusters[query_type]['queries']) > 50:
            self.query_clusters[query_type]['queries'].pop(0)
            
        # Update tool fingerprints with these keywords
        for tool_name, fingerprint in self.tool_fingerprints.items():
            # If tool has been used successfully for this query type before
            if tool_name in self.tool_history.get(query_type, {}) and \
               self.tool_history[query_type][tool_name]['success_rate'] > 0.6:
                fingerprint['usage_contexts'].add(query_type)
    
    def _check_cache(self, query: str, query_type: str) -> Union[str, List[str], None]:
        """
        Check if we have a cached result for a similar query.
        
        Args:
            query: The user query
            query_type: The type of query
            
        Returns:
            A cached tool selection or None
        """
        if not self.use_learning or query_type not in self.tool_history:
            return None
            
        # Find the most successful tool for this query type
        type_history = self.tool_history[query_type]
        best_tools = sorted(
            [(tool, data['success_rate']) for tool, data in type_history.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Only use cache if we have a high confidence result
        if best_tools and best_tools[0][1] > 0.75:
            return best_tools[0][0]
            
        return None
    
    def _cache_result(self, query: str, query_type: str, action: Union[str, List[str]]):
        """
        Cache a result for future similar queries.
        
        Args:
            query: The user query
            query_type: The type of query
            action: The selected action or plan
        """
        # Store in simulation cache
        query_key = self._get_query_cache_key(query)
        self.simulation_cache[query_key] = {
            'action': action,
            'timestamp': self._get_timestamp(),
            'query_type': query_type
        }
        
        # Limit cache size
        if len(self.simulation_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self.simulation_cache.keys(), 
                           key=lambda k: self.simulation_cache[k]['timestamp'])
            del self.simulation_cache[oldest_key]
    
    def _get_query_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        # In a real implementation, this might use a hash of query embeddings
        # For now, use a simple keyword approach
        keywords = ' '.join(sorted(set(word.lower() for word in query.split() if len(word) > 3)))
        return keywords[:100]  # Limit key length
    
    def _get_timestamp(self):
        """Get current timestamp."""
        import time
        return time.time()
    
    def _evaluate_state(self, state: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a state based on response quality and steps.
        
        Args:
            state: The current state
            
        Returns:
            A quality score
        """
        # Base score is the response quality
        score = state['response_quality']
        
        # If plan generation is enabled, we want to encourage complete plans
        if self.enable_plan_generation:
            steps_completed = len(state['actions_taken'])
            total_steps = steps_completed + state['steps_remaining']
            
            # Add bonus for completing more steps
            if total_steps > 0:
                step_completion_bonus = steps_completed / total_steps
                score += step_completion_bonus * 0.2  # 20% bonus for step completion
        
        return score
    
    def _generate_actions(self, state: Dict[str, Any]) -> List[str]:
        """
        Generate possible tool actions from the current state with intelligent filtering.
        
        Args:
            state: The current state
            
        Returns:
            List of possible actions
        """
        # Get query type
        query_type = state['query_type']
        query = state['query']
        
        # Get all available tools
        all_tools = set(self.tool_registry.get_all_tool_names())
        
        # Tools already used in this sequence
        used_tools = set(state['actions_taken'])
        
        # Remaining tools
        remaining_tools = all_tools - used_tools
        
        # If we're using learning, prioritize tools based on history
        if self.use_learning and query_type in self.tool_history:
            prioritized_tools = []
            
            # First, add tools that have been successful for this query type
            type_history = self.tool_history[query_type]
            
            # Check for successful tools
            for tool in remaining_tools:
                if tool in type_history and type_history[tool]['success_rate'] > 0.5:
                    prioritized_tools.append(tool)
                    
            # If we have at least some tools, return them
            if prioritized_tools and random.random() < self.tool_history_weight:
                return prioritized_tools
        
        # If using semantic similarity, filter by relevant tools
        if self.use_semantic_similarity:
            query_keywords = set(word.lower() for word in query.split() if len(word) > 3)
            
            # Score tools by semantic similarity to query
            scored_tools = []
            for tool in remaining_tools:
                if tool in self.tool_fingerprints:
                    fingerprint = self.tool_fingerprints[tool]
                    
                    # Calculate keyword overlap
                    keyword_overlap = len(query_keywords.intersection(fingerprint['keywords']))
                    
                    # Check if tool has been used for this query type
                    context_match = 1.0 if query_type in fingerprint['usage_contexts'] else 0.0
                    
                    # Combined score
                    score = keyword_overlap * 0.7 + context_match * 0.3
                    
                    scored_tools.append((tool, score))
            
            # Sort and filter tools
            scored_tools.sort(key=lambda x: x[1], reverse=True)
            
            # Take top half of tools if we have enough
            if len(scored_tools) > 2:
                return [t[0] for t in scored_tools[:max(2, len(scored_tools) // 2)]]
        
        # If we reach here, use all remaining tools
        return list(remaining_tools)
    
    def _simulate_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """
        Simulate taking an action (using a tool) in the given state with enhanced modeling.
        
        Args:
            state: The current state
            action: The tool action to simulate
            
        Returns:
            The new state after taking the action
        """
        # Create a new state with the action added
        new_state = state.copy()
        new_actions = state['actions_taken'].copy()
        new_actions.append(action)
        new_state['actions_taken'] = new_actions
        
        # Decrement steps remaining if using plan generation
        if self.enable_plan_generation and new_state['steps_remaining'] > 0:
            new_state['steps_remaining'] -= 1
        
        # Get query type and query
        query_type = state['query_type']
        query = state['query']
        
        # Simulate step result
        step_results = state['step_results'].copy()
        step_results[action] = self._simulate_tool_result(action, query)
        new_state['step_results'] = step_results
        
        # Estimate tool relevance based on learning or semantic similarity
        tool_relevance = self._estimate_tool_relevance(action, query, query_type)
        
        # Check for sequence effects (tools that work well together)
        sequence_bonus = 0.0
        if len(new_actions) > 1:
            prev_tool = new_actions[-2]
            sequence_key = f"{prev_tool}->{action}"
            if sequence_key in self.sequence_effectiveness:
                sequence_bonus = self.sequence_effectiveness[sequence_key] * 0.3  # 30% weight for sequence effects
        
        # Update quality based on relevance and sequence effects
        current_quality = state['response_quality']
        quality_improvement = tool_relevance + sequence_bonus
        
        # Add diminishing returns effect for additional tools
        if len(new_actions) > 1:
            diminishing_factor = 1.0 / len(new_actions)
            quality_improvement *= diminishing_factor
        
        new_quality = min(1.0, current_quality + quality_improvement)
        new_state['response_quality'] = new_quality
        
        return new_state
    
    def _simulate_tool_result(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Simulate the result of using a tool for a query.
        
        Args:
            tool_name: The name of the tool
            query: The user query
            
        Returns:
            A simulated result
        """
        # In a real implementation, this would be a more sophisticated simulation
        return {
            "tool": tool_name,
            "success_probability": self._estimate_tool_relevance(tool_name, query),
            "simulated": True
        }
    
    def _estimate_tool_relevance(self, tool_name: str, query: str, query_type: str = None) -> float:
        """
        Estimate how relevant a tool is for a given query using history and semantics.
        
        Args:
            tool_name: The name of the tool
            query: The user query
            query_type: Optional query type
            
        Returns:
            A relevance score between 0.0 and 1.0
        """
        relevance_score = 0.0
        
        # If we have historical data for this query type
        if self.use_learning and query_type and query_type in self.tool_history and \
           tool_name in self.tool_history[query_type]:
            
            # Get historical success rate
            history_score = self.tool_history[query_type][tool_name]['success_rate']
            relevance_score += history_score * self.tool_history_weight
        
        # If we're using semantic similarity
        if self.use_semantic_similarity and tool_name in self.tool_fingerprints:
            fingerprint = self.tool_fingerprints[tool_name]
            
            # Calculate keyword overlap
            query_keywords = set(word.lower() for word in query.split() if len(word) > 3)
            keyword_overlap = len(query_keywords.intersection(fingerprint['keywords']))
            
            # Normalize by query keywords
            if query_keywords:
                semantic_score = keyword_overlap / len(query_keywords)
                relevance_score += semantic_score * (1.0 - self.tool_history_weight)
        
        # Ensure we have a minimum score for exploration
        if relevance_score < 0.1:
            relevance_score = 0.1 + (random.random() * 0.1)  # Random boost between 0.1-0.2
        
        return relevance_score
    
    def _extract_plan_from_search(self) -> List[str]:
        """
        Extract a complete plan (tool sequence) from the search results.
        
        Returns:
            A list of tool names representing the plan
        """
        # In a real implementation, this would extract the highest value path 
        # from the search tree. For now, return None to indicate no plan extraction.
        return None
    
    def update_tool_history(self, tool_name: str, query: str, success: bool, 
                          execution_time: float, result: Any = None):
        """
        Update the tool history with the results of using a tool.
        
        Args:
            tool_name: The name of the tool used
            query: The query the tool was used for
            success: Whether the tool was successful
            execution_time: The execution time in seconds
            result: Optional result of the tool execution
        """
        if not self.use_learning:
            return
            
        # Get query type
        query_type = self._analyze_query(query)
        
        # Initialize history entry if needed
        if query_type not in self.tool_history:
            self.tool_history[query_type] = {}
            
        if tool_name not in self.tool_history[query_type]:
            self.tool_history[query_type][tool_name] = {
                'success_count': 0,
                'failure_count': 0,
                'total_time': 0.0,
                'success_rate': 0.0,
                'avg_time': 0.0,
                'examples': []
            }
        
        # Update history
        history = self.tool_history[query_type][tool_name]
        
        # Update counts
        if success:
            history['success_count'] += 1
        else:
            history['failure_count'] += 1
            
        # Update time
        history['total_time'] += execution_time
        
        # Update success rate
        total = history['success_count'] + history['failure_count']
        history['success_rate'] = history['success_count'] / total if total > 0 else 0.0
        
        # Update average time
        history['avg_time'] = history['total_time'] / total if total > 0 else 0.0
        
        # Add example (limit to last 5)
        history['examples'].append({
            'query': query,
            'success': success,
            'timestamp': self._get_timestamp()
        })
        if len(history['examples']) > 5:
            history['examples'].pop(0)
            
        # Update tool fingerprint
        if self.use_semantic_similarity and tool_name in self.tool_fingerprints:
            if success:
                # Add query type to usage contexts
                self.tool_fingerprints[tool_name]['usage_contexts'].add(query_type)
                
                # Add query keywords to tool fingerprint (with decay)
                query_keywords = set(word.lower() for word in query.split() if len(word) > 3)
                current_keywords = self.tool_fingerprints[tool_name]['keywords']
                
                # Add new keywords with adaptation rate
                for keyword in query_keywords:
                    if keyword not in current_keywords:
                        if random.random() < self.adaptation_rate:
                            current_keywords.add(keyword)
    
    def update_sequence_effectiveness(self, tool_sequence: List[str], success: bool, quality_score: float):
        """
        Update the effectiveness record for a sequence of tools.
        
        Args:
            tool_sequence: The sequence of tools used
            success: Whether the sequence was successful
            quality_score: A quality score for the sequence
        """
        if not self.use_learning or len(tool_sequence) < 2:
            return
            
        # Update pairwise effectiveness
        for i in range(len(tool_sequence) - 1):
            first_tool = tool_sequence[i]
            second_tool = tool_sequence[i + 1]
            sequence_key = f"{first_tool}->{second_tool}"
            
            if sequence_key not in self.sequence_effectiveness:
                self.sequence_effectiveness[sequence_key] = 0.5  # Initial neutral score
                
            # Update score with decay
            current_score = self.sequence_effectiveness[sequence_key]
            if success:
                # Increase score with quality bonus
                new_score = current_score + self.adaptation_rate * quality_score
            else:
                # Decrease score
                new_score = current_score - self.adaptation_rate
                
            # Clamp between 0 and 1
            self.sequence_effectiveness[sequence_key] = max(0.0, min(1.0, new_score))