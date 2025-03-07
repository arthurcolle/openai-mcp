from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="claude_code",
    version="0.1.0",
    author="Claude Code Team",
    author_email="noreply@example.com",
    description="Python recreation of Claude Code with enhanced features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/claude-code-python",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: User Interfaces",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "claude-code=claude_code.claude:app",
        ],
    },
)