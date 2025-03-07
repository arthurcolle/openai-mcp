#!/usr/bin/env python3
# claude_code/lib/ui/tool_visualizer.py
"""Real-time tool execution visualization."""

import logging
import time
import json
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.box import ROUNDED
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax

from ..tools.base import ToolResult

logger = logging.getLogger(__name__)


class ToolCallVisualizer:
    """Visualizes tool calls in real-time."""
    
    def __init__(self, console: Console):
        """Initialize the tool call visualizer.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        self.completed_calls: List[Dict[str, Any]] = []
        self.layout = self._create_layout()
        self.live = Live(self.layout, console=console, refresh_per_second=4, auto_refresh=False)
        self.max_completed_calls = 5
        
        # Keep track of recent tool results for routines
        self.recent_tool_results: List[ToolResult] = []
        self.max_recent_results = 20  # Maximum number of recent results to track
        
    def _create_layout(self) -> Layout:
        """Create the layout for the tool call visualization.
        
        Returns:
            Layout object
        """
        layout = Layout()
        layout.split(
            Layout(name="active", size=3),
            Layout(name="completed", size=3)
        )
        return layout
    
    def _create_active_calls_panel(self) -> Panel:
        """Create a panel with active tool calls.
        
        Returns:
            Panel with active call information
        """
        if not self.active_calls:
            return Panel(
                "No active tool calls",
                title="[bold blue]Active Tool Calls[/bold blue]",
                border_style="blue",
                box=ROUNDED
            )
        
        # Create progress bars for each active call
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            expand=True,
            console=self.console
        )
        
        # Add tasks for each active call
        for call_id, call_info in self.active_calls.items():
            if "task_id" not in call_info:
                # Create a new task for this call
                description = f"{call_info['tool_name']} ({call_id[:6]}...)"
                task_id = progress.add_task(description, total=100, completed=int(call_info["progress"] * 100))
                call_info["task_id"] = task_id
            else:
                # Update existing task
                progress.update(call_info["task_id"], completed=int(call_info["progress"] * 100))
        
        # Create a table with parameter information
        table = Table(show_header=True, header_style="bold cyan", box=ROUNDED, expand=True)
        table.add_column("Tool")
        table.add_column("Parameters")
        
        for call_id, call_info in self.active_calls.items():
            # Format parameters nicely
            params = call_info.get("parameters", {})
            if params:
                formatted_params = "\n".join([f"{k}: {self._format_value(v)}" for k, v in params.items()])
            else:
                formatted_params = "None"
            
            table.add_row(call_info["tool_name"], formatted_params)
        
        return Panel(
            progress,
            title="[bold blue]Active Tool Calls[/bold blue]",
            border_style="blue",
            box=ROUNDED
        )
    
    def _create_completed_calls_panel(self) -> Panel:
        """Create a panel with completed tool calls.
        
        Returns:
            Panel with completed call information
        """
        if not self.completed_calls:
            return Panel(
                "No completed tool calls",
                title="[bold green]Recent Tool Results[/bold green]",
                border_style="green",
                box=ROUNDED
            )
        
        # Create a table for results
        table = Table(show_header=True, header_style="bold green", box=ROUNDED, expand=True)
        table.add_column("Tool")
        table.add_column("Status")
        table.add_column("Time")
        table.add_column("Result Preview")
        
        # Show only the most recent completed calls
        for call_info in self.completed_calls[-self.max_completed_calls:]:
            tool_name = call_info["tool_name"]
            status = call_info["status"]
            execution_time = f"{call_info['execution_time']:.2f}s"
            
            # Format result preview
            result = call_info.get("result", "")
            if result:
                # Truncate and format result
                preview = self._format_result_preview(result, tool_name)
            else:
                preview = "No result"
            
            # Status with color
            status_text = Text(status)
            if status == "success":
                status_text.stylize("bold green")
            else:
                status_text.stylize("bold red")
            
            table.add_row(tool_name, status_text, execution_time, preview)
        
        return Panel(
            table,
            title="[bold green]Recent Tool Results[/bold green]",
            border_style="green",
            box=ROUNDED
        )
    
    def _format_value(self, value: Any) -> str:
        """Format a parameter value for display.
        
        Args:
            value: Parameter value
            
        Returns:
            Formatted string
        """
        if isinstance(value, (dict, list)):
            # Convert complex structures to JSON with indentation
            return json.dumps(value, indent=2)
        return str(value)
    
    def _format_result_preview(self, result: str, tool_name: str) -> str:
        """Format a result preview.
        
        Args:
            result: Result string
            tool_name: Name of the tool
            
        Returns:
            Formatted preview string
        """
        # Truncate result for preview
        if len(result) > 200:
            preview = result[:200] + "..."
        else:
            preview = result
        
        # Clean up newlines for display
        preview = preview.replace("\n", "\\n")
        
        return preview
    
    def start(self) -> None:
        """Start the visualization."""
        self.live.start()
        self.refresh()
    
    def stop(self) -> None:
        """Stop the visualization."""
        self.live.stop()
    
    def refresh(self) -> None:
        """Refresh the visualization."""
        # Update the layout with current information
        self.layout["active"].update(self._create_active_calls_panel())
        self.layout["completed"].update(self._create_completed_calls_panel())
        
        # Refresh the live display
        self.live.refresh()
    
    def add_tool_call(self, tool_call_id: str, tool_name: str, parameters: Dict[str, Any]) -> None:
        """Add a new tool call to visualize.
        
        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            parameters: Tool parameters
        """
        self.active_calls[tool_call_id] = {
            "tool_name": tool_name,
            "parameters": parameters,
            "start_time": time.time(),
            "progress": 0.0
        }
        self.refresh()
    
    def update_progress(self, tool_call_id: str, progress: float) -> None:
        """Update the progress of a tool call.
        
        Args:
            tool_call_id: ID of the tool call
            progress: Progress value (0-1)
        """
        if tool_call_id in self.active_calls:
            self.active_calls[tool_call_id]["progress"] = progress
            self.refresh()
    
    def complete_tool_call(self, tool_call_id: str, result: ToolResult) -> None:
        """Mark a tool call as complete.
        
        Args:
            tool_call_id: ID of the tool call
            result: Tool execution result
        """
        if tool_call_id in self.active_calls:
            call_info = self.active_calls[tool_call_id].copy()
            
            # Add result information
            call_info["result"] = result.result
            call_info["status"] = result.status
            call_info["execution_time"] = result.execution_time
            call_info["end_time"] = time.time()
            
            # Add to completed calls
            self.completed_calls.append(call_info)
            
            # Trim completed calls if needed
            if len(self.completed_calls) > self.max_completed_calls * 2:
                self.completed_calls = self.completed_calls[-self.max_completed_calls:]
            
            # Remove from active calls
            del self.active_calls[tool_call_id]
            
            # Store in recent tool results for routines
            if result.status == "success":
                self.recent_tool_results.append(result)
                # Keep only the most recent results
                if len(self.recent_tool_results) > self.max_recent_results:
                    self.recent_tool_results.pop(0)
            
            self.refresh()
    
    def show_result_detail(self, result: ToolResult) -> None:
        """Display detailed result information.
        
        Args:
            result: Tool execution result
        """
        # Detect if result might be code
        content = result.result
        if content.startswith(("def ", "class ", "import ", "from ")) or "```" in content:
            # Try to extract code blocks
            if "```" in content:
                blocks = content.split("```")
                # Find a code block with a language specifier
                for i in range(1, len(blocks), 2):
                    if i < len(blocks):
                        lang = blocks[i].split("\n")[0].strip()
                        code = "\n".join(blocks[i].split("\n")[1:])
                        if lang and code:
                            # Attempt to display as syntax-highlighted code
                            try:
                                syntax = Syntax(code, lang, theme="monokai", line_numbers=True)
                                self.console.print(Panel(syntax, title=f"[bold]Result: {result.name}[/bold]"))
                                return
                            except Exception:
                                pass
            
            # If we can't extract a code block, try to detect language
            for lang in ["python", "javascript", "bash", "json"]:
                try:
                    syntax = Syntax(content, lang, theme="monokai", line_numbers=True)
                    self.console.print(Panel(syntax, title=f"[bold]Result: {result.name}[/bold]"))
                    return
                except Exception:
                    pass
        
        # Just print as regular text if not code or if highlighting failed
        self.console.print(Panel(content, title=f"[bold]Result: {result.name}[/bold]"))


class MCTSVisualizer:
    """Visualizes the Monte Carlo Tree Search process in real-time with enhanced intelligence."""
    
    def __init__(self, console: Console):
        """Initialize the MCTS visualizer.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self.root_node = None
        self.current_iteration = 0
        self.max_iterations = 0
        self.best_action = None
        self.active_simulation = None
        self.simulation_path = []
        self.layout = self._create_layout()
        self.live = Live(self.layout, console=console, refresh_per_second=10, auto_refresh=False)
        
        # Intelligence enhancement - track history
        self.action_history = {}  # Track action performance over time
        self.visit_distribution = {}  # Track how visits are distributed
        self.exploration_patterns = []  # Track exploration patterns
        self.quality_metrics = {"search_efficiency": 0.0, "exploration_balance": 0.0}
        self.auto_improvement_enabled = True
        
    def _create_layout(self) -> Layout:
        """Create the layout for MCTS visualization.
        
        Returns:
            Layout object
        """
        layout = Layout()
        
        # Create the main sections with more detailed visualization
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="intelligence", size=7),  # New section for intelligence metrics
            Layout(name="stats", size=5)
        )
        
        # Split the main section into tree, simulation and action insights
        layout["main"].split_row(
            Layout(name="tree", ratio=2),
            Layout(name="simulation", ratio=1),
            Layout(name="insights", ratio=1)  # New section for action insights
        )
        
        return layout
        
    def set_search_parameters(self, root_node: Any, max_iterations: int, additional_params: Dict[str, Any] = None) -> None:
        """Set the search parameters with enhanced intelligence options.
        
        Args:
            root_node: The root node of the search tree
            max_iterations: Maximum number of iterations
            additional_params: Additional parameters for intelligent search
        """
        self.root_node = root_node
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
        # Initialize intelligence tracking
        self.action_history = {}
        self.visit_distribution = {}
        self.exploration_patterns = []
        
        # Set additional intelligence parameters
        if additional_params:
            self.auto_improvement_enabled = additional_params.get('auto_improvement', True)
            
            # Apply any initial intelligence strategies
            if additional_params.get('initial_action_bias'):
                self.action_history = additional_params['initial_action_bias']
                
        self.refresh()
        
    def update_iteration(self, iteration: int, selected_node: Any = None, 
                        expanded_node: Any = None, simulation_path: List[Any] = None,
                        simulation_result: float = None, best_action: Any = None,
                        node_values: Dict[str, float] = None) -> None:
        """Update the current iteration status with enhanced tracking.
        
        Args:
            iteration: Current iteration number
            selected_node: Node selected in this iteration
            expanded_node: Node expanded in this iteration
            simulation_path: Path of the simulation
            simulation_result: Result of the simulation
            best_action: Current best action
            node_values: Values of important nodes in the search (for visualization)
        """
        self.current_iteration = iteration
        self.selected_node = selected_node
        self.expanded_node = expanded_node
        self.simulation_path = simulation_path or []
        self.simulation_result = simulation_result
        
        if best_action is not None:
            self.best_action = best_action
            
        # Intelligence tracking - update action history
        if self.simulation_path and simulation_result is not None:
            for _, action in self.simulation_path:
                if action is not None:
                    action_str = str(action)
                    if action_str not in self.action_history:
                        self.action_history[action_str] = {
                            "visits": 0,
                            "total_value": 0.0,
                            "iterations": []
                        }
                    
                    self.action_history[action_str]["visits"] += 1
                    self.action_history[action_str]["total_value"] += simulation_result
                    self.action_history[action_str]["iterations"].append(iteration)
        
        # Update exploration pattern
        if selected_node:
            # Record exploration choice
            self.exploration_patterns.append({
                "iteration": iteration,
                "node_depth": self._get_node_depth(selected_node),
                "node_breadth": len(getattr(selected_node, "children", {})),
                "value_estimate": getattr(selected_node, "value", 0) / max(1, getattr(selected_node, "visits", 1))
            })
            
        # Update visit distribution
        if self.root_node and hasattr(self.root_node, "children"):
            self._update_visit_distribution()
            
        # Update quality metrics
        self._update_quality_metrics()
            
        self.refresh()
        
    def start(self) -> None:
        """Start the visualization."""
        self.live.start()
        self.refresh()
        
    def stop(self) -> None:
        """Stop the visualization."""
        self.live.stop()
        
    def refresh(self) -> None:
        """Refresh the visualization."""
        # Update header
        header_content = f"[bold blue]Enhanced Monte Carlo Tree Search - Iteration {self.current_iteration}/{self.max_iterations}[/bold blue]"
        if self.best_action:
            header_content += f" | Best Action: {self.best_action}"
            
        intelligence_status = "[green]Enabled[/green]" if self.auto_improvement_enabled else "[yellow]Disabled[/yellow]"
        header_content += f" | Intelligent Search: {intelligence_status}"
            
        self.layout["header"].update(Panel(header_content, border_style="blue"))
        
        # Update tree visualization
        self.layout["tree"].update(self._create_tree_panel())
        
        # Update simulation visualization
        self.layout["simulation"].update(self._create_simulation_panel())
        
        # Update action insights panel
        self.layout["insights"].update(self._create_insights_panel())
        
        # Update intelligence metrics
        self.layout["intelligence"].update(self._create_intelligence_panel())
        
        # Update stats
        self.layout["stats"].update(self._create_stats_panel())
        
        # Refresh the live display
        self.live.refresh()
        
    def _create_tree_panel(self) -> Panel:
        """Create a panel showing the current state of the search tree.
        
        Returns:
            Panel with tree visualization
        """
        if not self.root_node:
            return Panel("No search tree initialized", title="[bold]Search Tree[/bold]")
            
        # Create a table to show the tree structure
        from rich.tree import Tree
        from rich import box
        
        tree = Tree("🔍 Root Node", guide_style="bold blue")
        
        # Limit the depth and breadth for display
        max_depth = 3
        max_children = 5
        
        def add_node(node, tree_node, depth=0, path=None):
            if depth >= max_depth or not node or not hasattr(node, "children"):
                return
                
            if path is None:
                path = []
                
            # Add children nodes
            children = list(node.children.items())
            if not children:
                return
                
            # Sort children by a combination of visits and value
            def node_score(node_pair):
                child_node = node_pair[1]
                visits = getattr(child_node, "visits", 0)
                value = getattr(child_node, "value", 0)
                
                # Combine visits and value for scoring
                if visits > 0:
                    # Use UCB-style formula for ranking
                    exploitation = value / visits
                    exploration = (2 * 0.5 * (math.log(node.visits) / visits)) if node.visits > 0 and visits > 0 else 0
                    return exploitation + exploration
                return 0
            
            # Sort by this smarter formula
            children.sort(key=node_score, reverse=True)
            children = children[:max_children]
            
            for action, child in children:
                # Format node information
                visits = getattr(child, "visits", 0)
                value = getattr(child, "value", 0)
                
                # Highlight the node with more sophisticated coloring
                style = ""
                if child == self.selected_node:
                    style = "bold yellow"
                elif child == self.expanded_node:
                    style = "bold green"
                else:
                    # Color based on value
                    if visits > 0:
                        avg_value = value / visits
                        if avg_value > 0.7:
                            style = "green"
                        elif avg_value > 0.4:
                            style = "blue"
                        elif avg_value > 0.2:
                            style = "yellow"
                        else:
                            style = "red"
                
                # Create the node label with enhanced information
                current_path = path + [action]
                
                if visits > 0:
                    avg_value = value / visits
                    confidence = min(1.0, math.sqrt(visits) / 5) * 100  # Simple confidence estimate
                    label = f"[{style}]{action}: (Visits: {visits}, Value: {avg_value:.3f}, Conf: {confidence:.0f}%)[/{style}]"
                else:
                    label = f"[{style}]{action}: (New)[/{style}]"
                
                # Add the child node to the tree
                child_tree = tree_node.add(label)
                
                # Recursively add its children
                add_node(child, child_tree, depth + 1, current_path)
        
        # Start building the tree from the root
        if hasattr(self.root_node, "children"):
            # Add math import for node scoring
            import math
            add_node(self.root_node, tree)
            
        return Panel(tree, title="[bold]Search Tree[/bold]", border_style="blue")
    
    def _create_simulation_panel(self) -> Panel:
        """Create a panel showing the current simulation with enhanced analytics.
        
        Returns:
            Panel with simulation visualization
        """
        if not self.simulation_path:
            return Panel("No active simulation", title="[bold]Current Simulation[/bold]")
            
        # Create a list of simulation steps
        from rich.table import Table
        
        table = Table(box=None, expand=True)
        table.add_column("Step")
        table.add_column("Action")
        table.add_column("Expected Value")  # New column
        
        for i, (state, action) in enumerate(self.simulation_path):
            # Get expected value for this action
            action_str = str(action) if action is not None else "None"
            expected_value = "N/A"
            
            if action_str in self.action_history:
                history = self.action_history[action_str]
                if history["visits"] > 0:
                    expected_value = f"{history['total_value'] / history['visits']:.3f}"
            
            table.add_row(f"Step {i+1}", f"{action}", expected_value)
            
        if self.simulation_result is not None:
            # Add path quality metric
            path_quality = "Low"
            if self.simulation_result > 0.7:
                path_quality = "[bold green]High[/bold green]"
            elif self.simulation_result > 0.4:
                path_quality = "[yellow]Medium[/yellow]"
            else:
                path_quality = "[red]Low[/red]"
                
            table.add_row("Result", 
                        f"[bold green]{self.simulation_result:.3f}[/bold green]", 
                        f"Path Quality: {path_quality}")
            
        return Panel(table, title="[bold]Current Simulation[/bold]", border_style="green")
    
    def _create_insights_panel(self) -> Panel:
        """Create a panel showing action insights from learned patterns.
        
        Returns:
            Panel with action insights
        """
        from rich.table import Table
        
        if not self.action_history:
            return Panel("No action insights available yet", title="[bold]Action Insights[/bold]")
            
        # Get top performing actions
        top_actions = []
        for action, data in self.action_history.items():
            if data["visits"] >= 3:  # Only consider actions with enough samples
                avg_value = data["total_value"] / data["visits"]
                top_actions.append((action, avg_value, data["visits"]))
                
        # Sort by value and take top 5
        top_actions.sort(key=lambda x: x[1], reverse=True)
        top_actions = top_actions[:5]
        
        # Create insights table
        table = Table(box=None, expand=True)
        table.add_column("Action")
        table.add_column("Avg Value")
        table.add_column("Visits")
        table.add_column("Trend")
        
        for action, avg_value, visits in top_actions:
            # Generate trend indicator based on recent performance
            trend = "→"
            history = self.action_history[action]["iterations"]
            if len(history) >= 5:
                recent = set(history[-3:])  # Last 3 iterations
                if self.current_iteration - max(recent) <= 5:
                    trend = "↑"  # Recently used
                elif self.current_iteration - max(recent) >= 10:
                    trend = "↓"  # Not used recently
            
            # Color code based on value
            if avg_value > 0.7:
                value_str = f"[green]{avg_value:.3f}[/green]"
            elif avg_value > 0.4:
                value_str = f"[blue]{avg_value:.3f}[/blue]"
            else:
                value_str = f"[yellow]{avg_value:.3f}[/yellow]"
                
            table.add_row(str(action), value_str, str(visits), trend)
            
        return Panel(table, title="[bold]Action Insights[/bold]", border_style="cyan")
    
    def _create_intelligence_panel(self) -> Panel:
        """Create a panel showing intelligence metrics and learning patterns.
        
        Returns:
            Panel with intelligence visualization
        """
        from rich.table import Table
        from rich.columns import Columns
        
        # Create metrics table
        metrics_table = Table(box=None, expand=True)
        metrics_table.add_column("Metric")
        metrics_table.add_column("Value")
        
        # Add search quality metrics
        for metric, value in self.quality_metrics.items():
            formatted_name = metric.replace("_", " ").title()
            # Color based on value
            if value > 0.7:
                value_str = f"[green]{value:.2f}[/green]"
            elif value > 0.4:
                value_str = f"[blue]{value:.2f}[/blue]"
            else:
                value_str = f"[yellow]{value:.2f}[/yellow]"
                
            metrics_table.add_row(formatted_name, value_str)
            
        # Create exploration table
        exploration_table = Table(box=None, expand=True)
        exploration_table.add_column("Pattern")
        exploration_table.add_column("Value")
        
        # Add exploration patterns
        if self.exploration_patterns:
            # Average depth of exploration
            avg_depth = sum(p["node_depth"] for p in self.exploration_patterns) / len(self.exploration_patterns)
            exploration_table.add_row("Avg Exploration Depth", f"{avg_depth:.2f}")
            
            # Depth trend (increasing or decreasing)
            if len(self.exploration_patterns) >= 5:
                recent_avg = sum(p["node_depth"] for p in self.exploration_patterns[-5:]) / 5
                earlier_avg = sum(p["node_depth"] for p in self.exploration_patterns[:-5]) / max(1, len(self.exploration_patterns) - 5)
                
                if recent_avg > earlier_avg * 1.2:
                    trend = "[green]Deepening[/green]"
                elif recent_avg < earlier_avg * 0.8:
                    trend = "[yellow]Shallowing[/yellow]"
                else:
                    trend = "[blue]Stable[/blue]"
                    
                exploration_table.add_row("Depth Trend", trend)
                
            # Exploration-exploitation balance
            if len(self.exploration_patterns) >= 3:
                # Higher values = more exploitation of known good paths
                exploitation_ratio = sum(1 for p in self.exploration_patterns[-10:] 
                                     if p["value_estimate"] > 0.5) / min(10, len(self.exploration_patterns))
                
                if exploitation_ratio > 0.7:
                    balance = "[yellow]Heavy Exploitation[/yellow]"
                elif exploitation_ratio < 0.3:
                    balance = "[yellow]Heavy Exploration[/yellow]"
                else:
                    balance = "[green]Balanced[/green]"
                    
                exploration_table.add_row("Search Balance", balance)
                
        # Combine tables into columns
        columns = Columns([metrics_table, exploration_table])
        
        return Panel(columns, title="[bold]Intelligence Metrics[/bold]", border_style="magenta")
    
    def _create_stats_panel(self) -> Panel:
        """Create a panel showing search statistics with enhanced metrics.
        
        Returns:
            Panel with statistics
        """
        if not self.root_node:
            return Panel("No statistics available", title="[bold]Search Statistics[/bold]")
            
        # Collect statistics
        total_nodes = 0
        max_depth = 0
        total_visits = getattr(self.root_node, "visits", 0)
        avg_branching = 0
        
        def count_nodes(node, depth=0):
            nonlocal total_nodes, max_depth, avg_branching
            if not node or not hasattr(node, "children"):
                return
                
            total_nodes += 1
            max_depth = max(max_depth, depth)
            
            # Count children for branching factor
            num_children = len(node.children)
            if num_children > 0:
                avg_branching += num_children
                
            for child in node.children.values():
                count_nodes(child, depth + 1)
                
        count_nodes(self.root_node)
        
        # Calculate average branching factor
        if total_nodes > 1:  # Root node doesn't count for avg branching
            avg_branching /= (total_nodes - 1) 
        
        # Create a table of statistics
        from rich.table import Table
        
        table = Table(box=None, expand=True)
        table.add_column("Metric")
        table.add_column("Value")
        
        table.add_row("Total Nodes", str(total_nodes))
        table.add_row("Max Depth", str(max_depth))
        table.add_row("Total Visits", str(total_visits))
        table.add_row("Avg Branching", f"{avg_branching:.2f}")
        table.add_row("Progress", f"{self.current_iteration / self.max_iterations:.1%}")
        
        # Efficiency estimate (higher is better)
        if total_visits > 0:
            visit_efficiency = total_nodes / total_visits
            efficiency_str = f"{visit_efficiency:.2f}"
            table.add_row("Search Efficiency", efficiency_str)
        
        return Panel(table, title="[bold]Search Statistics[/bold]", border_style="magenta")
    
    def _get_node_depth(self, node):
        """Calculate the depth of a node in the tree."""
        depth = 0
        current = node
        while getattr(current, "parent", None) is not None:
            depth += 1
            current = current.parent
        return depth
    
    def _update_visit_distribution(self):
        """Update the distribution of visits across the tree."""
        levels = {}
        
        def count_visits_by_level(node, depth=0):
            if not node or not hasattr(node, "children"):
                return
                
            # Initialize level if not present
            if depth not in levels:
                levels[depth] = {"visits": 0, "nodes": 0}
                
            # Update level stats
            levels[depth]["visits"] += getattr(node, "visits", 0)
            levels[depth]["nodes"] += 1
            
            # Process children
            for child in node.children.values():
                count_visits_by_level(child, depth + 1)
                
        # Start counting from root
        count_visits_by_level(self.root_node)
        
        # Update visit distribution
        self.visit_distribution = levels
    
    def _update_quality_metrics(self):
        """Update quality metrics for the search process."""
        # Search efficiency - ratio of valuable nodes to total nodes
        # Higher values indicate more efficient search
        if self.visit_distribution:
            useful_visits = sum(level["visits"] for depth, level in self.visit_distribution.items() 
                               if depth > 0)  # Exclude root
            total_visits = sum(level["visits"] for level in self.visit_distribution.values())
            
            if total_visits > 0:
                self.quality_metrics["search_efficiency"] = useful_visits / total_visits
            
        # Exploration balance - how well the algorithm balances exploration vs exploitation
        if self.exploration_patterns:
            # Calculate variance in exploration depth
            depths = [p["node_depth"] for p in self.exploration_patterns[-20:]]  # Last 20 iterations
            if depths:
                import statistics
                try:
                    depth_variance = statistics.variance(depths) if len(depths) > 1 else 0
                    # Normalize to 0-1 range (higher variance = more balanced exploration)
                    normalized_variance = min(1.0, depth_variance / 5.0)  # Assume variance > 5 is high
                    self.quality_metrics["exploration_balance"] = normalized_variance
                except statistics.StatisticsError:
                    pass


class ParallelExecutionVisualizer:
    """Visualizes parallel execution of tool calls in real-time."""
    
    def __init__(self, console: Console):
        """Initialize the parallel execution visualizer.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self.active_executions = {}
        self.completed_executions = []
        self.layout = self._create_layout()
        self.live = Live(self.layout, console=console, refresh_per_second=10, auto_refresh=False)
        
    def _create_layout(self) -> Layout:
        """Create the layout for parallel execution visualization.
        
        Returns:
            Layout object
        """
        layout = Layout()
        
        # Create the main sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="executions"),
            Layout(name="metrics", size=5)
        )
        
        return layout
        
    def add_execution(self, execution_id: str, tool_name: str, parameters: Dict[str, Any]) -> None:
        """Add a new execution to visualize.
        
        Args:
            execution_id: Unique ID for the execution
            tool_name: Name of the tool being executed
            parameters: Parameters for the execution
        """
        self.active_executions[execution_id] = {
            "tool_name": tool_name,
            "parameters": parameters,
            "start_time": time.time(),
            "progress": 0.0,
            "status": "running"
        }
        self.refresh()
        
    def update_progress(self, execution_id: str, progress: float) -> None:
        """Update the progress of an execution.
        
        Args:
            execution_id: ID of the execution
            progress: Progress value (0-1)
        """
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["progress"] = progress
            self.refresh()
            
    def complete_execution(self, execution_id: str, result: Any, status: str = "success") -> None:
        """Mark an execution as complete.
        
        Args:
            execution_id: ID of the execution
            result: Result of the execution
            status: Status of completion
        """
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id].copy()
            execution["end_time"] = time.time()
            execution["duration"] = execution["end_time"] - execution["start_time"]
            execution["result"] = result
            execution["status"] = status
            
            # Move to completed executions
            self.completed_executions.append(execution)
            del self.active_executions[execution_id]
            
            # Limit completed executions list
            if len(self.completed_executions) > 20:
                self.completed_executions = self.completed_executions[-20:]
                
            self.refresh()
            
    def start(self) -> None:
        """Start the visualization."""
        self.live.start()
        self.refresh()
        
    def stop(self) -> None:
        """Stop the visualization."""
        self.live.stop()
        
    def refresh(self) -> None:
        """Refresh the visualization."""
        # Update header
        header_content = f"[bold blue]Parallel Execution Monitor[/bold blue] | Active: {len(self.active_executions)} | Completed: {len(self.completed_executions)}"
        self.layout["header"].update(Panel(header_content, border_style="blue"))
        
        # Update executions visualization
        self.layout["executions"].update(self._create_executions_panel())
        
        # Update metrics
        self.layout["metrics"].update(self._create_metrics_panel())
        
        # Refresh the live display
        self.live.refresh()
        
    def _create_executions_panel(self) -> Panel:
        """Create a panel showing active and recent executions.
        
        Returns:
            Panel with executions visualization
        """
        from rich.table import Table
        from rich.progress import BarColumn, Progress, TextColumn
        
        # Create progress bars for active executions
        progress_group = Table.grid(expand=True)
        
        if self.active_executions:
            # Create a progress group
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("| Elapsed: {task.elapsed:.2f}s"),
                expand=True
            )
            
            # Add tasks for each active execution
            for exec_id, execution in self.active_executions.items():
                tool_name = execution["tool_name"]
                description = f"{tool_name} ({exec_id[:8]}...)"
                task_id = progress.add_task(description, total=100, completed=int(execution["progress"] * 100))
                
            progress_group.add_row(progress)
        else:
            progress_group.add_row("[italic]No active executions[/italic]")
            
        # Create a table for completed executions
        completed_table = Table(show_header=True, header_style="bold blue", expand=True)
        completed_table.add_column("Tool")
        completed_table.add_column("Duration")
        completed_table.add_column("Status")
        completed_table.add_column("Result Preview")
        
        if self.completed_executions:
            # Most recent first
            for execution in reversed(self.completed_executions[-10:]):
                tool_name = execution["tool_name"]
                duration = f"{execution['duration']:.2f}s"
                status = execution["status"]
                
                # Format result preview
                result = str(execution.get("result", ""))
                preview = result[:50] + "..." if len(result) > 50 else result
                
                # Add status with color
                status_text = f"[green]{status}[/green]" if status == "success" else f"[red]{status}[/red]"
                
                completed_table.add_row(tool_name, duration, status_text, preview)
        else:
            completed_table.add_row("[italic]No completed executions[/italic]", "", "", "")
            
        # Combine both into a layout
        layout = Layout()
        layout.split(
            Layout(name="active", size=len(self.active_executions) * 2 + 3 if self.active_executions else 3),
            Layout(name="completed")
        )
        layout["active"].update(Panel(progress_group, title="[bold]Active Executions[/bold]", border_style="blue"))
        layout["completed"].update(Panel(completed_table, title="[bold]Recent Completions[/bold]", border_style="green"))
        
        return layout
    
    def _create_metrics_panel(self) -> Panel:
        """Create a panel showing execution metrics.
        
        Returns:
            Panel with metrics visualization
        """
        from rich.table import Table
        
        # Calculate metrics
        total_executions = len(self.completed_executions)
        successful = sum(1 for e in self.completed_executions if e["status"] == "success")
        failed = total_executions - successful
        
        if total_executions > 0:
            success_rate = successful / total_executions
            avg_duration = sum(e["duration"] for e in self.completed_executions) / total_executions
        else:
            success_rate = 0
            avg_duration = 0
            
        # Create metrics table
        table = Table(box=None, expand=True)
        table.add_column("Metric")
        table.add_column("Value")
        
        table.add_row("Total Executions", str(total_executions))
        table.add_row("Success Rate", f"{success_rate:.1%}")
        table.add_row("Average Duration", f"{avg_duration:.2f}s")
        table.add_row("Current Parallelism", str(len(self.active_executions)))
        
        return Panel(table, title="[bold]Execution Metrics[/bold]", border_style="magenta")


class MultiPanelLayout:
    """Creates a multi-panel layout for the entire UI."""
    
    def __init__(self, console: Console):
        """Initialize the multi-panel layout.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self.layout = self._create_layout()
        self.live = Live(self.layout, console=console, refresh_per_second=4, auto_refresh=False)
        
    def _create_layout(self) -> Layout:
        """Create the main application layout.
        
        Returns:
            Layout object
        """
        layout = Layout()
        
        # Split into three main sections
        layout.split(
            Layout(name="conversation", ratio=3),
            Layout(name="tools", ratio=2),
            Layout(name="input", ratio=1)
        )
        
        # Further split the tools section
        layout["tools"].split_row(
            Layout(name="active_tools"),
            Layout(name="cost", size=30)
        )
        
        return layout
    
    def start(self) -> None:
        """Start the live display."""
        self.live.start()
    
    def stop(self) -> None:
        """Stop the live display."""
        self.live.stop()
    
    def refresh(self) -> None:
        """Refresh the display."""
        self.live.refresh()
    
    def update_section(self, section: str, content: Any) -> None:
        """Update a section of the layout.
        
        Args:
            section: Section name
            content: Content to display
        """
        if section in self.layout:
            self.layout[section].update(content)
            self.refresh()