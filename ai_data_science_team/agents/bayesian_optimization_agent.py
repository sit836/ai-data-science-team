"""
Thin wrapper around the refactored bayesian optimization modules.
Keeps the public import path stable:
from ai_data_science_team.agents.bayesian_optimization_agent import make_bayesian_optimization_agent, BayesianOptimizationAgent
"""
from ai_data_science_team.agents.bayesian_optimization.agent import (
    make_bayesian_optimization_agent,
    BayesianOptimizationAgent,
)
