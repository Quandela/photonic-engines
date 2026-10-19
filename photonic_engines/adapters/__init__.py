"""Bridges between the engines and frontends. Each sub-package imports its frontend lazily
so the core stays torch-only.

  perceval   circuit -> unitary converter + Perceval backend wrappers
  merlin     `TimeLoopLayer`, a MerLin-style trainable layer for loop devices
"""
