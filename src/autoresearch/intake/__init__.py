"""From a repository URL to a config that loads, without running any of the repository's code.

``scope`` clones and decides whether the harness can measure the repository at
all. ``derive`` reads what can be read off the tree and writes a draft and a
brief. ``propose`` is one model conversation, with read only tools over the
clone, that chooses the call and the inputs. ``render`` turns an accepted
proposal into config text that is parsed back before it is written.

Code from an arbitrary URL only ever executes inside a box: ``check`` is the
first step that imports it.
"""
