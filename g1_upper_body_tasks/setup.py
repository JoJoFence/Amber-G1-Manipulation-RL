"""Installation script for the G1 upper body tasks extension."""

from setuptools import setup, find_packages

setup(
    name="g1_tasks",
    version="0.1.0",
    author="Jonas Hansen",
    description="G1 Upper Body Manipulation Tasks for Isaac Lab",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "isaaclab>=1.0.0",
    ],
)
