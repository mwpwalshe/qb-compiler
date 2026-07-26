"""Adversarial / hostile-input hardening tests for qb-compiler.

Every test here throws a deliberately malformed, oversized, or path-hostile input at a
public surface (CLI command, ObservableGate API, or parser) and asserts the surface
degrades gracefully: a clean error message + nonzero exit (CLI) or a typed exception
(API), never a raw traceback, hang, or unbounded resource use.
"""
