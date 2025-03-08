#!/usr/bin/env python3
# claude_code/lib/tools/code_tools.py
"""Code analysis and manipulation tools."""

import os
import logging
import subprocess
import tempfile
import json
from typing import Dict, List, Optional, Any, Union
import ast
import re

from .base import tool, ToolRegistry

logger = logging.getLogger(__name__)


@tool(
    name="CodeAnalyze",
    description="Analyze code to extract structure, dependencies, and complexity metrics",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to analyze"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform",
                "enum": ["structure", "complexity", "dependencies", "all"],
                "default": "all"
            }
        },
        "required": ["file_path"]
    },
    category="code"
)
def analyze_code(file_path: str, analysis_type: str = "all") -> str:
    """Analyze code to extract structure and metrics.
    
    Args:
        file_path: Path to the file to analyze
        analysis_type: Type of analysis to perform
        
    Returns:
        Analysis results as formatted text
    """
    logger.info(f"Analyzing code in {file_path} (type: {analysis_type})")
    
    if not os.path.isabs(file_path):
        return f"Error: File path must be absolute: {file_path}"
    
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Determine language
        if ext in ['.py']:
            return _analyze_python(code, analysis_type)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return _analyze_javascript(code, analysis_type)
        elif ext in ['.java']:
            return _analyze_java(code, analysis_type)
        elif ext in ['.c', '.cpp', '.cc', '.h', '.hpp']:
            return _analyze_cpp(code, analysis_type)
        else:
            return _analyze_generic(code, analysis_type)
    
    except Exception as e:
        logger.exception(f"Error analyzing code: {str(e)}")
        return f"Error analyzing code: {str(e)}"


def _analyze_python(code: str, analysis_type: str) -> str:
    """Analyze Python code."""
    result = []
    
    # Structure analysis
    if analysis_type in ["structure", "all"]:
        try:
            tree = ast.parse(code)
            
            # Extract classes
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if classes:
                result.append("Classes:")
                for cls in classes:
                    methods = [node.name for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)]
                    result.append(f"  - {cls.name}")
                    if methods:
                        result.append("    Methods:")
                        for method in methods:
                            result.append(f"      - {method}")
            
            # Extract functions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and 
                         not any(isinstance(parent, ast.ClassDef) for parent in ast.iter_child_nodes(tree))]
            if functions:
                result.append("\nFunctions:")
                for func in functions:
                    result.append(f"  - {func.name}")
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        imports.append(f"{module}.{name.name}")
            
            if imports:
                result.append("\nImports:")
                for imp in imports:
                    result.append(f"  - {imp}")
        
        except SyntaxError as e:
            result.append(f"Error parsing Python code: {str(e)}")
    
    # Complexity analysis
    if analysis_type in ["complexity", "all"]:
        try:
            # Count lines of code
            lines = code.count('\n') + 1
            non_empty_lines = sum(1 for line in code.split('\n') if line.strip())
            comment_lines = sum(1 for line in code.split('\n') if line.strip().startswith('#'))
            
            result.append("\nComplexity Metrics:")
            result.append(f"  - Total lines: {lines}")
            result.append(f"  - Non-empty lines: {non_empty_lines}")
            result.append(f"  - Comment lines: {comment_lines}")
            result.append(f"  - Code lines: {non_empty_lines - comment_lines}")
            
            # Cyclomatic complexity (simplified)
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.comprehension)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                    complexity += len(node.values) - 1
            
            result.append(f"  - Cyclomatic complexity (estimated): {complexity}")
        
        except Exception as e:
            result.append(f"Error calculating complexity: {str(e)}")
    
    # Dependencies analysis
    if analysis_type in ["dependencies", "all"]:
        try:
            # Extract imports
            tree = ast.parse(code)
            std_lib_imports = []
            third_party_imports = []
            local_imports = []
            
            std_lib_modules = [
                "abc", "argparse", "ast", "asyncio", "base64", "collections", "concurrent", "contextlib",
                "copy", "csv", "datetime", "decimal", "enum", "functools", "glob", "gzip", "hashlib",
                "http", "io", "itertools", "json", "logging", "math", "multiprocessing", "os", "pathlib",
                "pickle", "random", "re", "shutil", "socket", "sqlite3", "string", "subprocess", "sys",
                "tempfile", "threading", "time", "typing", "unittest", "urllib", "uuid", "xml", "zipfile"
            ]
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        module = name.name.split('.')[0]
                        if module in std_lib_modules:
                            std_lib_imports.append(name.name)
                        else:
                            third_party_imports.append(name.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module in std_lib_modules:
                            for name in node.names:
                                std_lib_imports.append(f"{node.module}.{name.name}")
                        elif node.level > 0:  # Relative import
                            for name in node.names:
                                local_imports.append(f"{'.' * node.level}{node.module or ''}.{name.name}")
                        else:
                            for name in node.names:
                                third_party_imports.append(f"{node.module}.{name.name}")
            
            result.append("\nDependencies:")
            if std_lib_imports:
                result.append("  Standard Library:")
                for imp in sorted(set(std_lib_imports)):
                    result.append(f"    - {imp}")
            
            if third_party_imports:
                result.append("  Third-Party:")
                for imp in sorted(set(third_party_imports)):
                    result.append(f"    - {imp}")
            
            if local_imports:
                result.append("  Local/Project:")
                for imp in sorted(set(local_imports)):
                    result.append(f"    - {imp}")
        
        except Exception as e:
            result.append(f"Error analyzing dependencies: {str(e)}")
    
    return "\n".join(result)


def _analyze_javascript(code: str, analysis_type: str) -> str:
    """Analyze JavaScript/TypeScript code."""
    result = []
    
    # Structure analysis
    if analysis_type in ["structure", "all"]:
        try:
            # Extract functions using regex (simplified)
            function_pattern = r'(function\s+(\w+)|const\s+(\w+)\s*=\s*function|const\s+(\w+)\s*=\s*\(.*?\)\s*=>)'
            functions = re.findall(function_pattern, code)
            
            if functions:
                result.append("Functions:")
                for func in functions:
                    # Get the first non-empty group which is the function name
                    func_name = next((name for name in func[1:] if name), "anonymous")
                    result.append(f"  - {func_name}")
            
            # Extract classes
            class_pattern = r'class\s+(\w+)'
            classes = re.findall(class_pattern, code)
            
            if classes:
                result.append("\nClasses:")
                for cls in classes:
                    result.append(f"  - {cls}")
            
            # Extract imports
            import_pattern = r'import\s+.*?from\s+[\'"](.+?)[\'"]'
            imports = re.findall(import_pattern, code)
            
            if imports:
                result.append("\nImports:")
                for imp in imports:
                    result.append(f"  - {imp}")
        
        except Exception as e:
            result.append(f"Error parsing JavaScript code: {str(e)}")
    
    # Complexity analysis
    if analysis_type in ["complexity", "all"]:
        try:
            # Count lines of code
            lines = code.count('\n') + 1
            non_empty_lines = sum(1 for line in code.split('\n') if line.strip())
            comment_lines = sum(1 for line in code.split('\n') 
                               if line.strip().startswith('//') or line.strip().startswith('/*'))
            
            result.append("\nComplexity Metrics:")
            result.append(f"  - Total lines: {lines}")
            result.append(f"  - Non-empty lines: {non_empty_lines}")
            result.append(f"  - Comment lines: {comment_lines}")
            result.append(f"  - Code lines: {non_empty_lines - comment_lines}")
            
            # Simplified cyclomatic complexity
            control_structures = len(re.findall(r'\b(if|for|while|switch|catch)\b', code))
            logical_operators = len(re.findall(r'(&&|\|\|)', code))
            
            complexity = 1 + control_structures + logical_operators
            result.append(f"  - Cyclomatic complexity (estimated): {complexity}")
        
        except Exception as e:
            result.append(f"Error calculating complexity: {str(e)}")
    
    # Dependencies analysis
    if analysis_type in ["dependencies", "all"]:
        try:
            # Extract imports
            import_pattern = r'import\s+.*?from\s+[\'"](.+?)[\'"]'
            imports = re.findall(import_pattern, code)
            
            node_std_libs = [
                "fs", "path", "http", "https", "url", "querystring", "crypto", "os", 
                "util", "stream", "events", "buffer", "assert", "zlib", "child_process"
            ]
            
            std_lib_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in imports:
                if imp in node_std_libs:
                    std_lib_imports.append(imp)
                elif imp.startswith('.'):
                    local_imports.append(imp)
                else:
                    third_party_imports.append(imp)
            
            result.append("\nDependencies:")
            if std_lib_imports:
                result.append("  Standard Library:")
                for imp in sorted(set(std_lib_imports)):
                    result.append(f"    - {imp}")
            
            if third_party_imports:
                result.append("  Third-Party:")
                for imp in sorted(set(third_party_imports)):
                    result.append(f"    - {imp}")
            
            if local_imports:
                result.append("  Local/Project:")
                for imp in sorted(set(local_imports)):
                    result.append(f"    - {imp}")
        
        except Exception as e:
            result.append(f"Error analyzing dependencies: {str(e)}")
    
    return "\n".join(result)


def _analyze_java(code: str, analysis_type: str) -> str:
    """Analyze Java code."""
    # Simplified Java analysis
    result = []
    
    # Structure analysis
    if analysis_type in ["structure", "all"]:
        try:
            # Extract class names
            class_pattern = r'(public|private|protected)?\s+class\s+(\w+)'
            classes = re.findall(class_pattern, code)
            
            if classes:
                result.append("Classes:")
                for cls in classes:
                    result.append(f"  - {cls[1]}")
            
            # Extract methods
            method_pattern = r'(public|private|protected)?\s+\w+\s+(\w+)\s*\([^)]*\)\s*\{'
            methods = re.findall(method_pattern, code)
            
            if methods:
                result.append("\nMethods:")
                for method in methods:
                    result.append(f"  - {method[1]}")
            
            # Extract imports
            import_pattern = r'import\s+(.+?);'
            imports = re.findall(import_pattern, code)
            
            if imports:
                result.append("\nImports:")
                for imp in imports:
                    result.append(f"  - {imp}")
        
        except Exception as e:
            result.append(f"Error parsing Java code: {str(e)}")
    
    # Complexity analysis
    if analysis_type in ["complexity", "all"]:
        try:
            # Count lines of code
            lines = code.count('\n') + 1
            non_empty_lines = sum(1 for line in code.split('\n') if line.strip())
            comment_lines = sum(1 for line in code.split('\n') 
                               if line.strip().startswith('//') or line.strip().startswith('/*'))
            
            result.append("\nComplexity Metrics:")
            result.append(f"  - Total lines: {lines}")
            result.append(f"  - Non-empty lines: {non_empty_lines}")
            result.append(f"  - Comment lines: {comment_lines}")
            result.append(f"  - Code lines: {non_empty_lines - comment_lines}")
            
            # Simplified cyclomatic complexity
            control_structures = len(re.findall(r'\b(if|for|while|switch|catch)\b', code))
            logical_operators = len(re.findall(r'(&&|\|\|)', code))
            
            complexity = 1 + control_structures + logical_operators
            result.append(f"  - Cyclomatic complexity (estimated): {complexity}")
        
        except Exception as e:
            result.append(f"Error calculating complexity: {str(e)}")
    
    return "\n".join(result)


def _analyze_cpp(code: str, analysis_type: str) -> str:
    """Analyze C/C++ code."""
    # Simplified C/C++ analysis
    result = []
    
    # Structure analysis
    if analysis_type in ["structure", "all"]:
        try:
            # Extract class names
            class_pattern = r'class\s+(\w+)'
            classes = re.findall(class_pattern, code)
            
            if classes:
                result.append("Classes:")
                for cls in classes:
                    result.append(f"  - {cls}")
            
            # Extract functions
            function_pattern = r'(\w+)\s+(\w+)\s*\([^)]*\)\s*\{'
            functions = re.findall(function_pattern, code)
            
            if functions:
                result.append("\nFunctions:")
                for func in functions:
                    # Filter out keywords that might be matched
                    if func[1] not in ['if', 'for', 'while', 'switch']:
                        result.append(f"  - {func[1]} (return type: {func[0]})")
            
            # Extract includes
            include_pattern = r'#include\s+[<"](.+?)[>"]'
            includes = re.findall(include_pattern, code)
            
            if includes:
                result.append("\nIncludes:")
                for inc in includes:
                    result.append(f"  - {inc}")
        
        except Exception as e:
            result.append(f"Error parsing C/C++ code: {str(e)}")
    
    # Complexity analysis
    if analysis_type in ["complexity", "all"]:
        try:
            # Count lines of code
            lines = code.count('\n') + 1
            non_empty_lines = sum(1 for line in code.split('\n') if line.strip())
            comment_lines = sum(1 for line in code.split('\n') 
                               if line.strip().startswith('//') or line.strip().startswith('/*'))
            
            result.append("\nComplexity Metrics:")
            result.append(f"  - Total lines: {lines}")
            result.append(f"  - Non-empty lines: {non_empty_lines}")
            result.append(f"  - Comment lines: {comment_lines}")
            result.append(f"  - Code lines: {non_empty_lines - comment_lines}")
            
            # Simplified cyclomatic complexity
            control_structures = len(re.findall(r'\b(if|for|while|switch|catch)\b', code))
            logical_operators = len(re.findall(r'(&&|\|\|)', code))
            
            complexity = 1 + control_structures + logical_operators
            result.append(f"  - Cyclomatic complexity (estimated): {complexity}")
        
        except Exception as e:
            result.append(f"Error calculating complexity: {str(e)}")
    
    return "\n".join(result)


def _analyze_generic(code: str, analysis_type: str) -> str:
    """Generic code analysis for unsupported languages."""
    result = []
    
    # Basic analysis for any language
    try:
        # Count lines of code
        lines = code.count('\n') + 1
        non_empty_lines = sum(1 for line in code.split('\n') if line.strip())
        
        result.append("Basic Code Metrics:")
        result.append(f"  - Total lines: {lines}")
        result.append(f"  - Non-empty lines: {non_empty_lines}")
        
        # Try to identify language
        language = "unknown"
        if "def " in code and "import " in code:
            language = "Python"
        elif "function " in code or "const " in code or "let " in code:
            language = "JavaScript"
        elif "public class " in code or "private class " in code:
            language = "Java"
        elif "#include" in code and "{" in code:
            language = "C/C++"
        
        result.append(f"  - Detected language: {language}")
        
        # Find potential functions/methods using a generic pattern
        function_pattern = r'\b(\w+)\s*\([^)]*\)\s*\{'
        functions = re.findall(function_pattern, code)
        
        if functions:
            result.append("\nPotential Functions/Methods:")
            for func in functions:
                # Filter out common keywords
                if func not in ['if', 'for', 'while', 'switch', 'catch']:
                    result.append(f"  - {func}")
    
    except Exception as e:
        result.append(f"Error analyzing code: {str(e)}")
    
    return "\n".join(result)


@tool(
    name="LintCode",
    description="Lint code to find potential issues and style violations",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to lint"
            },
            "linter": {
                "type": "string",
                "description": "Linter to use (auto, pylint, eslint, etc.)",
                "default": "auto"
            }
        },
        "required": ["file_path"]
    },
    category="code"
)
def lint_code(file_path: str, linter: str = "auto") -> str:
    """Lint code to find potential issues.
    
    Args:
        file_path: Path to the file to lint
        linter: Linter to use
        
    Returns:
        Linting results as formatted text
    """
    logger.info(f"Linting code in {file_path} using {linter}")
    
    if not os.path.isabs(file_path):
        return f"Error: File path must be absolute: {file_path}"
    
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Auto-detect linter if not specified
        if linter == "auto":
            if ext in ['.py']:
                linter = "pylint"
            elif ext in ['.js', '.jsx']:
                linter = "eslint"
            elif ext in ['.ts', '.tsx']:
                linter = "tslint"
            elif ext in ['.java']:
                linter = "checkstyle"
            elif ext in ['.c', '.cpp', '.cc', '.h', '.hpp']:
                linter = "cppcheck"
            else:
                return f"Error: Could not auto-detect linter for file type {ext}"
        
        # Run appropriate linter
        if linter == "pylint":
            return _run_pylint(file_path)
        elif linter == "eslint":
            return _run_eslint(file_path)
        elif linter == "tslint":
            return _run_tslint(file_path)
        elif linter == "checkstyle":
            return _run_checkstyle(file_path)
        elif linter == "cppcheck":
            return _run_cppcheck(file_path)
        else:
            return f"Error: Unsupported linter: {linter}"
    
    except Exception as e:
        logger.exception(f"Error linting code: {str(e)}")
        return f"Error linting code: {str(e)}"


def _run_pylint(file_path: str) -> str:
    """Run pylint on a Python file."""
    try:
        # Check if pylint is installed
        try:
            subprocess.run(["pylint", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Error: pylint is not installed. Please install it with 'pip install pylint'."
        
        # Run pylint
        result = subprocess.run(
            ["pylint", "--output-format=text", file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return "No issues found."
        
        # Format output
        output = result.stdout or result.stderr
        
        # Summarize output
        lines = output.split('\n')
        summary_lines = [line for line in lines if "rated at" in line]
        issue_lines = [line for line in lines if re.match(r'^.*?:\d+:\d+:', line)]
        
        formatted_output = []
        
        if issue_lines:
            formatted_output.append("Issues found:")
            for line in issue_lines:
                formatted_output.append(f"  {line}")
        
        if summary_lines:
            formatted_output.append("\nSummary:")
            for line in summary_lines:
                formatted_output.append(f"  {line}")
        
        return "\n".join(formatted_output)
    
    except Exception as e:
        return f"Error running pylint: {str(e)}"


def _run_eslint(file_path: str) -> str:
    """Run eslint on a JavaScript file."""
    try:
        # Check if eslint is installed
        try:
            subprocess.run(["eslint", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Error: eslint is not installed. Please install it with 'npm install -g eslint'."
        
        # Run eslint
        result = subprocess.run(
            ["eslint", "--format=stylish", file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return "No issues found."
        
        # Format output
        output = result.stdout or result.stderr
        
        # Clean up output
        lines = output.split('\n')
        filtered_lines = [line for line in lines if line.strip() and not line.startswith("eslint:")]
        
        return "\n".join(filtered_lines)
    
    except Exception as e:
        return f"Error running eslint: {str(e)}"


def _run_tslint(file_path: str) -> str:
    """Run tslint on a TypeScript file."""
    try:
        # Check if tslint is installed
        try:
            subprocess.run(["tslint", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Error: tslint is not installed. Please install it with 'npm install -g tslint'."
        
        # Run tslint
        result = subprocess.run(
            ["tslint", "-t", "verbose", file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return "No issues found."
        
        # Format output
        output = result.stdout or result.stderr
        
        return output
    
    except Exception as e:
        return f"Error running tslint: {str(e)}"


def _run_checkstyle(file_path: str) -> str:
    """Run checkstyle on a Java file."""
    return "Checkstyle support not implemented. Please install and run checkstyle manually."


def _run_cppcheck(file_path: str) -> str:
    """Run cppcheck on a C/C++ file."""
    try:
        # Check if cppcheck is installed
        try:
            subprocess.run(["cppcheck", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Error: cppcheck is not installed. Please install it using your system package manager."
        
        # Run cppcheck
        result = subprocess.run(
            ["cppcheck", "--enable=all", "--template='{file}:{line}: {severity}: {message}'", file_path],
            capture_output=True,
            text=True
        )
        
        # Format output
        output = result.stderr  # cppcheck outputs to stderr
        
        if not output or "no errors found" in output.lower():
            return "No issues found."
        
        # Clean up output
        lines = output.split('\n')
        filtered_lines = [line for line in lines if line.strip() and "Checking" not in line]
        
        return "\n".join(filtered_lines)
    
    except Exception as e:
        return f"Error running cppcheck: {str(e)}"


def register_code_tools(registry: ToolRegistry) -> None:
    """Register all code analysis tools with the registry.
    
    Args:
        registry: Tool registry to register with
    """
    from .base import create_tools_from_functions
    
    code_tools = [
        analyze_code,
        lint_code
    ]
    
    create_tools_from_functions(registry, code_tools)
