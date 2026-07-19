"""Baseline memory systems: they turn a Task's sessions into reader context.

A "system" is anything with build(sessions) -> object exposing
context_for(question) -> str. Baselines set the floor (naive RAG) and the
reference ceiling (full context); CogniTrace itself will plug into the same
interface in Phase 1.
"""
