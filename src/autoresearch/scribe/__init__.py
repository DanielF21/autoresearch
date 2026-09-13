"""The Scribe: reads a finished run and drafts the pull request for its best candidate.

Code filters the candidates. One model picks one of them, or none, and in the same
conversation reads the target repository's last merged pull requests by people and
writes the pull request in their style. It never writes into a run directory and
never opens a pull request.
"""
