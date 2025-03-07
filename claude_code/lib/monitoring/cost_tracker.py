#!/usr/bin/env python3
# claude_code/lib/monitoring/cost_tracker.py
"""Cost tracking and management."""

import logging
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks token usage and calculates costs for LLM interactions."""
    
    def __init__(self, budget_limit: Optional[float] = None, history_file: Optional[str] = None):
        """Initialize the cost tracker.
        
        Args:
            budget_limit: Optional budget limit in dollars
            history_file: Optional path to a file to store history
        """
        self.budget_limit = budget_limit
        self.history_file = history_file
        
        # Initialize session counters
        self.session_start = datetime.now()
        self.session_tokens_input = 0
        self.session_tokens_output = 0
        self.session_cost = 0.0
        
        # Request history
        self.requests: List[Dict[str, Any]] = []
        
        # Load history from file if provided
        self._load_history()
    
    def add_request(self, 
                   provider: str, 
                   model: str, 
                   tokens_input: int, 
                   tokens_output: int,
                   input_cost_per_1k: float,
                   output_cost_per_1k: float,
                   request_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a request to the tracker.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4o", "claude-3-opus")
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            input_cost_per_1k: Cost per 1,000 input tokens
            output_cost_per_1k: Cost per 1,000 output tokens
            request_id: Optional request ID
            
        Returns:
            Dictionary with request information including costs
        """
        # Calculate costs
        input_cost = (tokens_input / 1000) * input_cost_per_1k
        output_cost = (tokens_output / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        # Update session counters
        self.session_tokens_input += tokens_input
        self.session_tokens_output += tokens_output
        self.session_cost += total_cost
        
        # Create request record
        request = {
            "id": request_id or f"{int(time.time())}-{len(self.requests)}",
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
        
        # Add to history
        self.requests.append(request)
        
        # Save history
        self._save_history()
        
        # Log the request
        logger.info(
            f"Request: {provider}/{model}, " +
            f"Tokens: {tokens_input} in / {tokens_output} out, " +
            f"Cost: ${total_cost:.4f}"
        )
        
        return request
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session.
        
        Returns:
            Dictionary with session statistics
        """
        return {
            "start_time": self.session_start.isoformat(),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "tokens_input": self.session_tokens_input,
            "tokens_output": self.session_tokens_output,
            "total_tokens": self.session_tokens_input + self.session_tokens_output,
            "total_cost": self.session_cost,
            "request_count": len(self.requests),
            "budget_limit": self.budget_limit,
            "budget_remaining": None if self.budget_limit is None else self.budget_limit - self.session_cost
        }
    
    def check_budget(self) -> Dict[str, Any]:
        """Check if budget limit is approached or exceeded.
        
        Returns:
            Dictionary with budget status information
        """
        if self.budget_limit is None:
            return {
                "has_budget": False,
                "status": "no_limit",
                "message": "No budget limit set"
            }
        
        remaining = self.budget_limit - self.session_cost
        percentage_used = (self.session_cost / self.budget_limit) * 100
        
        if remaining <= 0:
            status = "exceeded"
            message = f"Budget exceeded by ${abs(remaining):.2f}"
        elif percentage_used > 90:
            status = "critical"
            message = f"Budget critical: ${remaining:.2f} remaining ({percentage_used:.1f}% used)"
        elif percentage_used > 75:
            status = "warning"
            message = f"Budget warning: ${remaining:.2f} remaining ({percentage_used:.1f}% used)"
        else:
            status = "ok"
            message = f"Budget OK: ${remaining:.2f} remaining ({percentage_used:.1f}% used)"
        
        return {
            "has_budget": True,
            "status": status,
            "message": message,
            "limit": self.budget_limit,
            "used": self.session_cost,
            "remaining": remaining,
            "percentage_used": percentage_used
        }
    
    def get_usage_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics grouped by model.
        
        Returns:
            Dictionary mapping "provider/model" to usage statistics
        """
        usage: Dict[str, Dict[str, Any]] = {}
        
        for request in self.requests:
            key = f"{request['provider']}/{request['model']}"
            
            if key not in usage:
                usage[key] = {
                    "provider": request["provider"],
                    "model": request["model"],
                    "request_count": 0,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "total_cost": 0.0
                }
            
            usage[key]["request_count"] += 1
            usage[key]["tokens_input"] += request["tokens_input"]
            usage[key]["tokens_output"] += request["tokens_output"]
            usage[key]["total_cost"] += request["total_cost"]
        
        return usage
    
    def get_cost_summary_panel(self) -> Panel:
        """Create a Rich panel with cost summary information.
        
        Returns:
            Rich Panel object
        """
        # Get stats and budget info
        stats = self.get_session_stats()
        budget = self.check_budget()
        
        # Create a table for the summary
        table = Table(show_header=False, box=ROUNDED, expand=True)
        table.add_column("Item", style="bold")
        table.add_column("Value")
        
        # Add rows with token usage
        table.add_row(
            "Tokens (Input)",
            f"{stats['tokens_input']:,}"
        )
        table.add_row(
            "Tokens (Output)",
            f"{stats['tokens_output']:,}"
        )
        table.add_row(
            "Total Cost",
            f"${stats['total_cost']:.4f}"
        )
        
        # Add budget information if available
        if budget["has_budget"]:
            # Create styled text for budget status
            status_text = Text(budget["message"])
            if budget["status"] == "exceeded":
                status_text.stylize("bold red")
            elif budget["status"] == "critical":
                status_text.stylize("bold yellow")
            elif budget["status"] == "warning":
                status_text.stylize("yellow")
            else:
                status_text.stylize("green")
            
            table.add_row("Budget", status_text)
        
        # Create the panel
        title = "[bold]Cost & Usage Summary[/bold]"
        return Panel(table, title=title, border_style="yellow")
    
    def reset_session(self) -> None:
        """Reset the session counters but keep request history."""
        self.session_start = datetime.now()
        self.session_tokens_input = 0
        self.session_tokens_output = 0
        self.session_cost = 0.0
        
        logger.info("Cost tracking session reset")
    
    def _save_history(self) -> None:
        """Save request history to file if configured."""
        if not self.history_file:
            return
        
        try:
            # Ensure directory exists
            directory = os.path.dirname(self.history_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Save history
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_start": self.session_start.isoformat(),
                    "budget_limit": self.budget_limit,
                    "requests": self.requests,
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cost history: {e}")
    
    def _load_history(self) -> None:
        """Load request history from file if available."""
        if not self.history_file or not os.path.exists(self.history_file):
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Load session data
                self.session_start = datetime.fromisoformat(data.get('session_start', self.session_start.isoformat()))
                self.budget_limit = data.get('budget_limit', self.budget_limit)
                
                # Load requests
                self.requests = data.get('requests', [])
                
                # Recalculate session totals
                self.session_tokens_input = sum(r.get('tokens_input', 0) for r in self.requests)
                self.session_tokens_output = sum(r.get('tokens_output', 0) for r in self.requests)
                self.session_cost = sum(r.get('total_cost', 0) for r in self.requests)
                
                logger.info(f"Loaded cost history with {len(self.requests)} requests")
        except Exception as e:
            logger.error(f"Failed to load cost history: {e}")
    
    def generate_usage_report(self, format: str = "text") -> str:
        """Generate a usage report.
        
        Args:
            format: Output format ("text", "json", "markdown")
            
        Returns:
            Formatted usage report
        """
        stats = self.get_session_stats()
        model_usage = self.get_usage_by_model()
        
        if format == "json":
            return json.dumps({
                "session": stats,
                "models": model_usage
            }, indent=2)
        
        # Text or markdown format
        lines = []
        lines.append("# Usage Report" if format == "markdown" else "USAGE REPORT")
        lines.append("")
        
        # Session summary
        lines.append("## Session Summary" if format == "markdown" else "SESSION SUMMARY")
        lines.append(f"- Start time: {stats['start_time']}")
        lines.append(f"- Duration: {stats['duration_seconds'] / 60:.1f} minutes")
        lines.append(f"- Requests: {stats['request_count']}")
        lines.append(f"- Total tokens: {stats['total_tokens']:,} ({stats['tokens_input']:,} in / {stats['tokens_output']:,} out)")
        lines.append(f"- Total cost: ${stats['total_cost']:.4f}")
        if stats['budget_limit'] is not None:
            lines.append(f"- Budget: ${stats['budget_limit']:.2f} (${stats['budget_remaining']:.2f} remaining)")
        lines.append("")
        
        # Usage by model
        lines.append("## Usage by Model" if format == "markdown" else "USAGE BY MODEL")
        for key, usage in sorted(model_usage.items(), key=lambda x: x[1]['total_cost'], reverse=True):
            lines.append(f"### {key}" if format == "markdown" else key.upper())
            lines.append(f"- Requests: {usage['request_count']}")
            lines.append(f"- Tokens: {usage['tokens_input'] + usage['tokens_output']:,} ({usage['tokens_input']:,} in / {usage['tokens_output']:,} out)")
            lines.append(f"- Cost: ${usage['total_cost']:.4f}")
            lines.append("")
        
        return "\n".join(lines)