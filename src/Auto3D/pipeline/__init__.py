"""Orchestration: what runs, in what order, in which process.

Currently one module. It is a package rather than a flat module because the
plan's later items move the stage sequence, the executors and the run context
here, and because ``input_checks`` needed a home above the model layer before
any of that -- it had been living in ``utils``, reaching upward.
"""
