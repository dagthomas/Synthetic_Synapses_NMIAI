"""Google ADK-based autonomous research agent for Astar Island prediction optimization.

Uses Gemini to propose structural algorithmic changes, backtests them against
ground truth data, and iterates to find improvements over the current best score.

Usage:
    python -m research_agent.run
    python -m research_agent.run --max-iters 10
    python -m research_agent.run --dry-run
"""
