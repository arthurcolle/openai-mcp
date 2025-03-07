#!/usr/bin/env python3
# claude_code/lib/tools/manager.py
"""Tool execution manager."""

import logging
import time
import json
import uuid
import os
from typing import Dict, List, Any, Optional, Callable, Union, Sequence
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future

from .base import Tool, ToolResult, ToolRegistry, Routine, RoutineStep, RoutineDefinition

logger = logging.getLogger(__name__)


class RoutineExecutionManager:
    """Manages the execution of tool routines."""
    
    def __init__(self, registry: ToolRegistry, execution_manager: 'ToolExecutionManager'):
        """Initialize the routine execution manager.
        
        Args:
            registry: Tool registry containing available tools and routines
            execution_manager: Tool execution manager for executing individual tools
        """
        self.registry = registry
        self.execution_manager = execution_manager
        self.active_routines: Dict[str, Dict[str, Any]] = {}
        self.progress_callback: Optional[Callable[[str, str, float], None]] = None
        self.result_callback: Optional[Callable[[str, List[ToolResult]], None]] = None
        
        # Load existing routines
        self.registry.load_routines()
    
    def set_progress_callback(self, callback: Callable[[str, str, float], None]) -> None:
        """Set a callback function for routine progress updates.
        
        Args:
            callback: Function that takes routine_id, step_name, and progress (0-1) as arguments
        """
        self.progress_callback = callback
    
    def set_result_callback(self, callback: Callable[[str, List[ToolResult]], None]) -> None:
        """Set a callback function for routine results.
        
        Args:
            callback: Function that takes routine_id and list of ToolResults as arguments
        """
        self.result_callback = callback
    
    def create_routine(self, definition: RoutineDefinition) -> str:
        """Create a new routine from a definition.
        
        Args:
            definition: Routine definition
            
        Returns:
            Routine ID
            
        Raises:
            ValueError: If a routine with the same name already exists
        """
        # Convert step objects to dictionaries
        steps = []
        for step in definition.steps:
            step_dict = {
                "tool_name": step.tool_name,
                "args": step.args
            }
            if step.condition is not None:
                step_dict["condition"] = step.condition
            if step.store_result:
                step_dict["store_result"] = True
                if step.result_var is not None:
                    step_dict["result_var"] = step.result_var
            
            steps.append(step_dict)
        
        # Create routine
        routine = Routine(
            name=definition.name,
            description=definition.description,
            steps=steps
        )
        
        # Register routine
        self.registry.register_routine(routine)
        
        return routine.name
    
    def create_routine_from_tool_history(
        self, 
        name: str, 
        description: str, 
        tool_results: List[ToolResult],
        context_variables: Dict[str, Any] = None
    ) -> str:
        """Create a routine from a history of tool executions.
        
        Args:
            name: Name for the routine
            description: Description of the routine
            tool_results: List of tool results to base the routine on
            context_variables: Optional dictionary of context variables to identify
            
        Returns:
            Routine ID
        """
        steps = []
        
        # Process tool results into steps
        for i, result in enumerate(tool_results):
            # Skip failed tool calls
            if result.status != "success":
                continue
            
            # Get tool
            tool = self.registry.get_tool(result.name)
            if not tool:
                continue
            
            # Extract arguments from tool call
            args = {}
            # Here we would need to extract the arguments from the tool call
            # This is a simplification and would need to be adapted to the actual structure
            
            # Create step
            step = {
                "tool_name": result.name,
                "args": args,
                "store_result": True,
                "result_var": f"result_{i}"
            }
            
            steps.append(step)
        
        # Create routine
        routine = Routine(
            name=name,
            description=description,
            steps=steps
        )
        
        # Register routine
        self.registry.register_routine(routine)
        
        return routine.name
    
    def execute_routine(self, name: str, context: Dict[str, Any] = None) -> str:
        """Execute a routine with the given context.
        
        Args:
            name: Name of the routine to execute
            context: Context variables for the routine
            
        Returns:
            Routine execution ID
            
        Raises:
            ValueError: If the routine is not found
        """
        # Get routine
        routine = self.registry.get_routine(name)
        if not routine:
            raise ValueError(f"Routine not found: {name}")
        
        # Create execution ID
        execution_id = str(uuid.uuid4())
        
        # Initialize context
        if context is None:
            context = {}
        
        # Initialize execution state
        self.active_routines[execution_id] = {
            "routine": routine,
            "context": context.copy(),
            "results": [],
            "current_step": 0,
            "start_time": time.time(),
            "status": "running"
        }
        
        # Record routine usage
        self.registry.record_routine_usage(name)
        
        # Start execution in background thread
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self._execute_routine_steps, execution_id)
        
        return execution_id
    
    def _execute_routine_steps(self, execution_id: str) -> None:
        """Execute the steps of a routine in sequence.
        
        Args:
            execution_id: Routine execution ID
        """
        if execution_id not in self.active_routines:
            logger.error(f"Routine execution not found: {execution_id}")
            return
        
        execution = self.active_routines[execution_id]
        routine = execution["routine"]
        context = execution["context"]
        results = execution["results"]
        
        try:
            # Execute each step
            for i, step in enumerate(routine.steps):
                # Update current step
                execution["current_step"] = i
                
                # Check for conditions
                if "condition" in step and not self._evaluate_condition(step["condition"], context, results):
                    logger.info(f"Skipping step {i+1}/{len(routine.steps)} due to condition")
                    continue
                
                # Process tool arguments with variable substitution
                processed_args = self._process_arguments(step["args"], context, results)
                
                # Create tool call
                tool_call = {
                    "id": f"{execution_id}_{i}",
                    "function": {
                        "name": step["tool_name"],
                        "arguments": json.dumps(processed_args)
                    }
                }
                
                # Report progress
                self._report_routine_progress(execution_id, i, len(routine.steps), step["tool_name"])
                
                # Execute tool
                result = self.execution_manager.execute_tool(tool_call)
                
                # Add result to results
                results.append(result)
                
                # Store result in context if requested
                if step.get("store_result", False):
                    var_name = step.get("result_var", f"result_{i}")
                    context[var_name] = result.result
                
                # Check for loop control
                if "repeat_until" in step and not self._evaluate_condition(step["repeat_until"], context, results):
                    # Go back to specified step
                    target_step = step.get("repeat_target", 0)
                    if 0 <= target_step < i:
                        i = target_step - 1  # Will be incremented in next loop iteration
                
                # Check for exit condition
                if "exit_condition" in step and self._evaluate_condition(step["exit_condition"], context, results):
                    logger.info(f"Exiting routine early due to exit condition at step {i+1}/{len(routine.steps)}")
                    break
            
            # Update execution status
            execution["status"] = "completed"
            
            # Report final progress
            self._report_routine_progress(execution_id, len(routine.steps), len(routine.steps), "completed")
            
            # Call result callback
            if self.result_callback:
                self.result_callback(execution_id, results)
                
        except Exception as e:
            logger.exception(f"Error executing routine: {e}")
            execution["status"] = "error"
            execution["error"] = str(e)
            
            # Report error progress
            self._report_routine_progress(execution_id, execution["current_step"], len(routine.steps), "error")
    
    def _process_arguments(
        self,
        args: Dict[str, Any],
        context: Dict[str, Any],
        results: List[ToolResult]
    ) -> Dict[str, Any]:
        """Process tool arguments with variable substitution.
        
        Args:
            args: Tool arguments
            context: Context variables
            results: Previous tool results
            
        Returns:
            Processed arguments
        """
        processed_args = {}
        
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                # Variable reference
                var_name = value[1:]
                if var_name in context:
                    processed_args[key] = context[var_name]
                elif var_name.startswith("result[") and var_name.endswith("]"):
                    # Reference to previous result
                    try:
                        idx = int(var_name[7:-1])
                        if 0 <= idx < len(results):
                            processed_args[key] = results[idx].result
                        else:
                            processed_args[key] = value
                    except (ValueError, IndexError):
                        processed_args[key] = value
                else:
                    processed_args[key] = value
            else:
                processed_args[key] = value
        
        return processed_args
    
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        context: Dict[str, Any],
        results: List[ToolResult]
    ) -> bool:
        """Evaluate a condition for a routine step.
        
        Args:
            condition: Condition specification
            context: Context variables
            results: Previous tool results
            
        Returns:
            Whether the condition is met
        """
        condition_type = condition.get("type", "simple")
        
        if condition_type == "simple":
            # Simple variable comparison
            var_name = condition.get("variable", "")
            operation = condition.get("operation", "equals")
            value = condition.get("value")
            
            # Get variable value
            var_value = None
            if var_name.startswith("$"):
                var_name = var_name[1:]
                var_value = context.get(var_name)
            elif var_name.startswith("result[") and var_name.endswith("]"):
                try:
                    idx = int(var_name[7:-1])
                    if 0 <= idx < len(results):
                        var_value = results[idx].result
                except (ValueError, IndexError):
                    return False
            
            # Compare
            if operation == "equals":
                return var_value == value
            elif operation == "not_equals":
                return var_value != value
            elif operation == "contains":
                return value in var_value if var_value is not None else False
            elif operation == "greater_than":
                return var_value > value if var_value is not None else False
            elif operation == "less_than":
                return var_value < value if var_value is not None else False
            
            return False
        
        elif condition_type == "and":
            # Logical AND of multiple conditions
            sub_conditions = condition.get("conditions", [])
            return all(self._evaluate_condition(c, context, results) for c in sub_conditions)
        
        elif condition_type == "or":
            # Logical OR of multiple conditions
            sub_conditions = condition.get("conditions", [])
            return any(self._evaluate_condition(c, context, results) for c in sub_conditions)
        
        elif condition_type == "not":
            # Logical NOT
            sub_condition = condition.get("condition", {})
            return not self._evaluate_condition(sub_condition, context, results)
        
        return True  # Default to True
    
    def _report_routine_progress(
        self,
        execution_id: str,
        current_step: int,
        total_steps: int,
        step_name: str
    ) -> None:
        """Report progress for a routine execution.
        
        Args:
            execution_id: Routine execution ID
            current_step: Current step index
            total_steps: Total number of steps
            step_name: Name of the current step
        """
        progress = current_step / total_steps if total_steps > 0 else 1.0
        
        # Call progress callback if set
        if self.progress_callback:
            self.progress_callback(execution_id, step_name, progress)
    
    def get_active_routines(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active routine executions.
        
        Returns:
            Dictionary mapping execution ID to routine execution information
        """
        return {
            k: {
                "routine_name": v["routine"].name,
                "current_step": v["current_step"],
                "total_steps": len(v["routine"].steps),
                "status": v["status"],
                "start_time": v["start_time"],
                "elapsed_time": time.time() - v["start_time"]
            }
            for k, v in self.active_routines.items()
        }
    
    def get_routine_results(self, execution_id: str) -> Optional[List[ToolResult]]:
        """Get the results of a routine execution.
        
        Args:
            execution_id: Routine execution ID
            
        Returns:
            List of tool results, or None if the routine execution is not found
        """
        if execution_id in self.active_routines:
            return self.active_routines[execution_id]["results"]
        return None
    
    def cancel_routine(self, execution_id: str) -> bool:
        """Cancel a routine execution.
        
        Args:
            execution_id: Routine execution ID
            
        Returns:
            Whether the routine was canceled successfully
        """
        if execution_id in self.active_routines:
            self.active_routines[execution_id]["status"] = "canceled"
            return True
        return False


class ToolExecutionManager:
    """Manages tool execution, including parallel execution and progress tracking."""
    
    def __init__(self, registry: ToolRegistry):
        """Initialize the tool execution manager.
        
        Args:
            registry: Tool registry containing available tools
        """
        self.registry = registry
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.progress_callback: Optional[Callable[[str, float], None]] = None
        self.result_callback: Optional[Callable[[str, ToolResult], None]] = None
        self.max_workers = 10
        
        # Initialize routine manager
        self.routine_manager = RoutineExecutionManager(registry, self)
    
    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set a callback function for progress updates.
        
        Args:
            callback: Function that takes tool_call_id and progress (0-1) as arguments
        """
        self.progress_callback = callback
    
    def set_result_callback(self, callback: Callable[[str, ToolResult], None]) -> None:
        """Set a callback function for results.
        
        Args:
            callback: Function that takes tool_call_id and ToolResult as arguments
        """
        self.result_callback = callback
    
    def execute_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a single tool synchronously.
        
        Args:
            tool_call: Dictionary containing tool call information
            
        Returns:
            ToolResult with execution result
            
        Raises:
            ValueError: If the tool is not found
        """
        function_name = tool_call.get("function", {}).get("name", "")
        tool_call_id = tool_call.get("id", "unknown")
        
        # Check if it's a routine
        if function_name.startswith("routine."):
            routine_name = function_name[9:]  # Remove "routine." prefix
            return self._execute_routine_as_tool(routine_name, tool_call)
        
        # Get the tool
        tool = self.registry.get_tool(function_name)
        if not tool:
            error_msg = f"Tool not found: {function_name}"
            logger.error(error_msg)
            return ToolResult(
                tool_call_id=tool_call_id,
                name=function_name,
                result=f"Error: {error_msg}",
                execution_time=0,
                status="error",
                error=error_msg
            )
        
        # Check if tool needs permission and handle it
        if tool.needs_permission:
            # TODO: Implement permission handling
            logger.warning(f"Tool {function_name} needs permission, but permission handling is not implemented")
        
        # Track progress
        self._track_execution_start(tool_call_id, function_name)
        
        try:
            # Execute the tool
            result = tool.execute(tool_call)
            
            # Track completion
            self._track_execution_complete(tool_call_id, result)
            
            return result
        except Exception as e:
            logger.exception(f"Error executing tool {function_name}: {e}")
            result = ToolResult(
                tool_call_id=tool_call_id,
                name=function_name,
                result=f"Error: {str(e)}",
                execution_time=0,
                status="error",
                error=str(e)
            )
            self._track_execution_complete(tool_call_id, result)
            return result
    
    def _execute_routine_as_tool(self, routine_name: str, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a routine as if it were a tool.
        
        Args:
            routine_name: Name of the routine
            tool_call: Dictionary containing tool call information
            
        Returns:
            ToolResult with execution result
        """
        tool_call_id = tool_call.get("id", "unknown")
        start_time = time.time()
        
        try:
            # Extract context from arguments
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                context = json.loads(arguments_str)
            except json.JSONDecodeError:
                context = {}
            
            # Execute routine
            execution_id = self.routine_manager.execute_routine(routine_name, context)
            
            # Wait for routine to complete
            while True:
                routine_status = self.routine_manager.get_active_routines().get(execution_id, {})
                if routine_status.get("status") != "running":
                    break
                time.sleep(0.1)
            
            # Get results
            results = self.routine_manager.get_routine_results(execution_id)
            if not results:
                raise ValueError(f"No results from routine: {routine_name}")
            
            # Format results
            result_summary = f"Routine {routine_name} executed successfully with {len(results)} steps\n\n"
            for i, result in enumerate(results):
                result_summary += f"Step {i+1}: {result.name} - {'SUCCESS' if result.status == 'success' else 'ERROR'}\n"
                if result.status != "success":
                    result_summary += f"  Error: {result.error}\n"
            
            # Track execution time
            execution_time = time.time() - start_time
            
            # Create result
            return ToolResult(
                tool_call_id=tool_call_id,
                name=f"routine.{routine_name}",
                result=result_summary,
                execution_time=execution_time,
                status="success"
            )
        
        except Exception as e:
            logger.exception(f"Error executing routine {routine_name}: {e}")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=f"routine.{routine_name}",
                result=f"Error: {str(e)}",
                execution_time=time.time() - start_time,
                status="error",
                error=str(e)
            )
    
    def execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tools in parallel.
        
        Args:
            tool_calls: List of dictionaries containing tool call information
            
        Returns:
            List of ToolResult with execution results
        """
        results = []
        futures: Dict[Future, str] = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tool_calls))) as executor:
            # Submit all tool calls
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id", "unknown")
                future = executor.submit(self.execute_tool, tool_call)
                futures[future] = tool_call_id
            
            # Wait for completion and collect results
            for future in concurrent.futures.as_completed(futures):
                tool_call_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.exception(f"Error in parallel tool execution for {tool_call_id}: {e}")
                    # Create an error result
                    function_name = next(
                        (tc.get("function", {}).get("name", "") for tc in tool_calls 
                         if tc.get("id", "") == tool_call_id), 
                        "unknown"
                    )
                    results.append(ToolResult(
                        tool_call_id=tool_call_id,
                        name=function_name,
                        result=f"Error: {str(e)}",
                        execution_time=0,
                        status="error",
                        error=str(e)
                    ))
        
        return results
    
    def create_routine(self, definition: RoutineDefinition) -> str:
        """Create a new routine.
        
        Args:
            definition: Routine definition
            
        Returns:
            Routine ID
        """
        return self.routine_manager.create_routine(definition)
    
    def create_routine_from_tool_history(
        self, 
        name: str, 
        description: str, 
        tool_results: List[ToolResult],
        context_variables: Dict[str, Any] = None
    ) -> str:
        """Create a routine from a history of tool executions.
        
        Args:
            name: Name for the routine
            description: Description of the routine
            tool_results: List of tool results to base the routine on
            context_variables: Optional dictionary of context variables to identify
            
        Returns:
            Routine ID
        """
        return self.routine_manager.create_routine_from_tool_history(
            name, description, tool_results, context_variables
        )
    
    def execute_routine(self, name: str, context: Dict[str, Any] = None) -> str:
        """Execute a routine with the given context.
        
        Args:
            name: Name of the routine to execute
            context: Context variables for the routine
            
        Returns:
            Routine execution ID
        """
        return self.routine_manager.execute_routine(name, context)
    
    def get_routine_results(self, execution_id: str) -> Optional[List[ToolResult]]:
        """Get the results of a routine execution.
        
        Args:
            execution_id: Routine execution ID
            
        Returns:
            List of tool results, or None if the routine execution is not found
        """
        return self.routine_manager.get_routine_results(execution_id)
    
    def _track_execution_start(self, tool_call_id: str, tool_name: str) -> None:
        """Track the start of tool execution.
        
        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
        """
        self.active_executions[tool_call_id] = {
            "tool_name": tool_name,
            "start_time": time.time(),
            "progress": 0.0
        }
        
        # Call progress callback if set
        if self.progress_callback:
            self.progress_callback(tool_call_id, 0.0)
    
    def _track_execution_progress(self, tool_call_id: str, progress: float) -> None:
        """Track the progress of tool execution.
        
        Args:
            tool_call_id: ID of the tool call
            progress: Progress value (0-1)
        """
        if tool_call_id in self.active_executions:
            self.active_executions[tool_call_id]["progress"] = progress
            
            # Call progress callback if set
            if self.progress_callback:
                self.progress_callback(tool_call_id, progress)
    
    def _track_execution_complete(self, tool_call_id: str, result: ToolResult) -> None:
        """Track the completion of tool execution.
        
        Args:
            tool_call_id: ID of the tool call
            result: Tool execution result
        """
        if tool_call_id in self.active_executions:
            # Update progress
            self._track_execution_progress(tool_call_id, 1.0)
            
            # Calculate execution time
            start_time = self.active_executions[tool_call_id]["start_time"]
            execution_time = time.time() - start_time
            
            # Clean up
            del self.active_executions[tool_call_id]
            
            # Call result callback if set
            if self.result_callback:
                self.result_callback(tool_call_id, result)
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active tool executions.
        
        Returns:
            Dictionary mapping tool_call_id to execution information
        """
        return self.active_executions.copy()
    
    def cancel_execution(self, tool_call_id: str) -> bool:
        """Cancel a tool execution if possible.
        
        Args:
            tool_call_id: ID of the tool call to cancel
            
        Returns:
            True if canceled successfully, False otherwise
        """
        # TODO: Implement cancellation logic
        # This would require more sophisticated execution tracking
        logger.warning(f"Cancellation not implemented for tool_call_id: {tool_call_id}")
        return False