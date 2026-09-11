"""Autoresearch: an agent harness that makes CPU bound Python faster and proves it.

Three stateless parts and one writer.

- A worker proposes one patch per attempt from inside its own sailbox.
- A referee judges one patch on its own sailbox: tests, timing, instruction count.
- An orchestrator runs rounds, calls workers and referees, and is the only thing that
  writes to the run directory.

Host code, everything outside ``guest``, runs on the laptop or the control box and may
import the Sail SDK. Guest code, under ``guest``, is uploaded into sailboxes and may
import only the standard library.
"""

__version__ = "0.2.0"
