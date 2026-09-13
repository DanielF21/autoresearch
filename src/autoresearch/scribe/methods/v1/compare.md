You choose which one of several performance patches to propose to a repository's maintainers. Each was already reviewed on its own and judged mergeable, and each was measured against the same base commit on the same inputs.

None of them is both faster and smaller than another, so the choice is a trade. Weigh what a maintainer weighs when reviewing: how much faster it is on every input, how much code they have to read and keep, how easy the change is to understand from the diff, whether it keeps an old code path alive next to the new one, and how much risk it carries for inputs the benchmark did not cover.

You can read each patched tree at the root named in its heading, and the base at `base:`. Call `submit_pick` with the attempt number and the reasons, stated in terms of the diffs and the measurements you were given.
