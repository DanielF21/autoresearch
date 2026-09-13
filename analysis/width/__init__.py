"""The width experiment: the same target at widths 1, 4, 16 and 64, read as one table.

``load`` turns each run directory into one row per attempt using the harness's own
readers. ``dims`` computes the seven dimensions of the analysis and writes one table
and one figure each. ``python -m analysis.width`` runs both.
"""
