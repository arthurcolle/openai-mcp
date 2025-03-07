#!/usr/bin/env python3
"""Module for tracking MCP server metrics."""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque, Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerMetrics:
    """Tracks MCP server metrics for visualization."""
    
    def __init__(self, history_size: int = 100, save_interval: int = 60):
        """Initialize the server metrics tracker.
        
        Args:
            history_size: Number of data points to keep in history
            save_interval: How often to save metrics to disk (in seconds)
        """
        self._start_time = time.time()
        self._lock = threading.RLock()
        self._history_size = history_size
        self._save_interval = save_interval
        self._save_path = os.path.expanduser("~/.config/claude_code/metrics.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
        
        # Metrics
        self._request_history = deque(maxlen=history_size)
        self._tool_calls = Counter()
        self._resource_calls = Counter()
        self._connections = 0
        self._active_connections = set()
        self._errors = Counter()
        
        # Time series data for charts
        self._time_series = {
            "tool_calls": deque([(time.time(), 0)] * 10, maxlen=10),
            "resource_calls": deque([(time.time(), 0)] * 10, maxlen=10)
        }
        
        # Start auto-save thread
        self._running = True
        self._save_thread = threading.Thread(target=self._auto_save, daemon=True)
        self._save_thread.start()
        
        # Load previous metrics if available
        self._load_metrics()
    
    def _auto_save(self):
        """Periodically save metrics to disk."""
        while self._running:
            time.sleep(self._save_interval)
            try:
                self.save_metrics()
            except Exception as e:
                logger.error(f"Error saving metrics: {e}")
    
    def _load_metrics(self):
        """Load metrics from disk if available."""
        try:
            if os.path.exists(self._save_path):
                with open(self._save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                with self._lock:
                    # Load previous tool and resource calls
                    self._tool_calls = Counter(data.get("tool_calls", {}))
                    self._resource_calls = Counter(data.get("resource_calls", {}))
                    
                    # Don't load time-sensitive data like connections and history
                    
                    logger.info(f"Loaded metrics from {self._save_path}")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def save_metrics(self):
        """Save metrics to disk."""
        try:
            with self._lock:
                data = {
                    "tool_calls": dict(self._tool_calls),
                    "resource_calls": dict(self._resource_calls),
                    "total_connections": self._connections,
                    "last_saved": time.time()
                }
            
            with open(self._save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Metrics saved to {self._save_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def log_tool_call(self, tool_name: str, success: bool = True):
        """Log a tool call.
        
        Args:
            tool_name: The name of the tool that was called
            success: Whether the call was successful
        """
        with self._lock:
            self._tool_calls[tool_name] += 1
            
            # Add to request history
            timestamp = time.time()
            self._request_history.append({
                "type": "tool",
                "name": tool_name,
                "success": success,
                "timestamp": timestamp
            })
            
            # Update time series
            current_time = time.time()
            last_time, count = self._time_series["tool_calls"][-1]
            if current_time - last_time < 60:  # Less than a minute
                self._time_series["tool_calls"][-1] = (last_time, count + 1)
            else:
                self._time_series["tool_calls"].append((current_time, 1))
    
    def log_resource_request(self, resource_uri: str, success: bool = True):
        """Log a resource request.
        
        Args:
            resource_uri: The URI of the requested resource
            success: Whether the request was successful
        """
        with self._lock:
            self._resource_calls[resource_uri] += 1
            
            # Add to request history
            timestamp = time.time()
            self._request_history.append({
                "type": "resource",
                "uri": resource_uri,
                "success": success,
                "timestamp": timestamp
            })
            
            # Update time series
            current_time = time.time()
            last_time, count = self._time_series["resource_calls"][-1]
            if current_time - last_time < 60:  # Less than a minute
                self._time_series["resource_calls"][-1] = (last_time, count + 1)
            else:
                self._time_series["resource_calls"].append((current_time, 1))
    
    def log_connection(self, client_id: str, connected: bool = True):
        """Log a client connection or disconnection.
        
        Args:
            client_id: Client identifier
            connected: True for connection, False for disconnection
        """
        with self._lock:
            if connected:
                self._connections += 1
                self._active_connections.add(client_id)
            else:
                self._active_connections.discard(client_id)
            
            # Add to request history
            timestamp = time.time()
            self._request_history.append({
                "type": "connection",
                "client_id": client_id,
                "action": "connect" if connected else "disconnect",
                "timestamp": timestamp
            })
    
    def log_error(self, error_type: str, message: str):
        """Log an error.
        
        Args:
            error_type: Type of error
            message: Error message
        """
        with self._lock:
            self._errors[error_type] += 1
            
            # Add to request history
            timestamp = time.time()
            self._request_history.append({
                "type": "error",
                "error_type": error_type,
                "message": message,
                "timestamp": timestamp
            })
    
    def get_uptime(self) -> str:
        """Get the server uptime as a human-readable string.
        
        Returns:
            Uptime string (e.g., "2 hours 15 minutes")
        """
        uptime_seconds = time.time() - self._start_time
        uptime = timedelta(seconds=int(uptime_seconds))
        
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} {'day' if days == 1 else 'days'}")
        if hours > 0 or days > 0:
            parts.append(f"{hours} {'hour' if hours == 1 else 'hours'}")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
        
        if not parts:
            return f"{seconds} seconds"
        
        return " ".join(parts)
    
    def get_active_connections_count(self) -> int:
        """Get the number of active connections.
        
        Returns:
            Number of active connections
        """
        with self._lock:
            return len(self._active_connections)
    
    def get_total_connections(self) -> int:
        """Get the total number of connections since startup.
        
        Returns:
            Total connection count
        """
        with self._lock:
            return self._connections
    
    def get_recent_activity(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent activity.
        
        Args:
            count: Number of recent events to return
            
        Returns:
            List of recent activity events
        """
        with self._lock:
            recent = list(self._request_history)[-count:]
            
            # Format timestamps
            for event in recent:
                ts = event["timestamp"]
                event["formatted_time"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            
            return recent
    
    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get statistics on tool usage.
        
        Returns:
            Dictionary mapping tool names to call counts
        """
        with self._lock:
            return dict(self._tool_calls)
    
    def get_resource_usage_stats(self) -> Dict[str, int]:
        """Get statistics on resource usage.
        
        Returns:
            Dictionary mapping resource URIs to request counts
        """
        with self._lock:
            return dict(self._resource_calls)
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get statistics on errors.
        
        Returns:
            Dictionary mapping error types to counts
        """
        with self._lock:
            return dict(self._errors)
    
    def get_time_series_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get time series data for charts.
        
        Returns:
            Dictionary with time series data
        """
        with self._lock:
            result = {}
            
            # Convert deques to lists of dictionaries
            for series_name, series_data in self._time_series.items():
                result[series_name] = [
                    {"timestamp": ts, "value": val, "formatted_time": datetime.fromtimestamp(ts).strftime("%H:%M:%S")}
                    for ts, val in series_data
                ]
            
            return result
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics data.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            "uptime": self.get_uptime(),
            "active_connections": self.get_active_connections_count(),
            "total_connections": self.get_total_connections(),
            "recent_activity": self.get_recent_activity(20),
            "tool_usage": self.get_tool_usage_stats(),
            "resource_usage": self.get_resource_usage_stats(),
            "errors": self.get_error_stats(),
            "time_series": self.get_time_series_data()
        }
    
    def reset_stats(self):
        """Reset all statistics but keep the start time."""
        with self._lock:
            self._request_history.clear()
            self._tool_calls.clear()
            self._resource_calls.clear()
            self._connections = 0
            self._active_connections.clear()
            self._errors.clear()
            
            # Reset time series
            current_time = time.time()
            self._time_series = {
                "tool_calls": deque([(current_time - (600 - i * 60), 0) for i in range(10)], maxlen=10),
                "resource_calls": deque([(current_time - (600 - i * 60), 0) for i in range(10)], maxlen=10)
            }
    
    def shutdown(self):
        """Shutdown the metrics tracker and save data."""
        self._running = False
        self.save_metrics()


# Singleton instance
_metrics_instance = None

def get_metrics() -> ServerMetrics:
    """Get or create the singleton metrics instance.
    
    Returns:
        ServerMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ServerMetrics()
    return _metrics_instance