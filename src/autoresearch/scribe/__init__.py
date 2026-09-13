"""The Scribe: reads a finished run, picks the candidate worth shipping, drafts its PR.

It never writes into a run directory, never creates a box, and never opens a PR.
Everything it produces goes under its own output root, and every model call it
makes is recorded next to what the call produced.

Two paths share the machinery here. ``write`` is production: a frozen method run
end to end with no human. ``dev`` is where that method is tuned, and nothing in
the production path imports it.
"""
