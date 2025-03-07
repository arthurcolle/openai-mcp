#!/bin/bash
# Installation script for Claude Code Python Edition

# Set up colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing Claude Code Python Edition...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${YELLOW}Detected Python version: ${python_version}${NC}"

# Check if Python version is at least 3.10
if [[ $(echo "${python_version}" | cut -d. -f1,2 | sed 's/\.//') -lt 310 ]]; then
    echo -e "${RED}Error: Python 3.10 or higher is required.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error creating virtual environment.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Using existing virtual environment.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Error activating virtual environment.${NC}"
    exit 1
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Error installing dependencies.${NC}"
    exit 1
fi

# Install in development mode
echo -e "${YELLOW}Installing Claude Code in development mode...${NC}"
pip install -e .
if [ $? -ne 0 ]; then
    echo -e "${RED}Error installing package.${NC}"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOF
# API Keys (uncomment and add your keys)
# OPENAI_API_KEY=your_openai_api_key
# ANTHROPIC_API_KEY=your_anthropic_api_key

# Models (optional)
# OPENAI_MODEL=gpt-4o
# ANTHROPIC_MODEL=claude-3-opus-20240229

# Budget limit in dollars (optional)
# BUDGET_LIMIT=5.0
EOF
    echo -e "${YELLOW}Created .env file. Please edit it to add your API keys.${NC}"
else
    echo -e "${YELLOW}.env file already exists. Skipping creation.${NC}"
fi

# Create setup.py if it doesn't exist
if [ ! -f "setup.py" ]; then
    echo -e "${YELLOW}Creating setup.py...${NC}"
    cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="claude_code",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt", "r").readlines()
    ],
    entry_points={
        "console_scripts": [
            "claude-code=claude_code.claude:app",
        ],
    },
)
EOF
    echo -e "${YELLOW}Created setup.py file.${NC}"
else
    echo -e "${YELLOW}setup.py file already exists. Skipping creation.${NC}"
fi

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
echo -e "    source venv/bin/activate"
echo -e "${YELLOW}To run Claude Code, use:${NC}"
echo -e "    claude-code"
echo -e "${YELLOW}Or:${NC}"
echo -e "    python -m claude_code.claude"
echo -e "${GREEN}Enjoy using Claude Code Python Edition!${NC}"