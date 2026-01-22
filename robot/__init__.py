"""
Robot Environment Module

Provides unified interface for robot control and sensing.
Supports both real robot and simulation (future).
"""

from .real_env import RealRobotEnv

# Alias for convenience
RobotEnv = RealRobotEnv

__all__ = [
    'RealRobotEnv',
    'RobotEnv',
]