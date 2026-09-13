"""The referee pool: one referee box per worker slot, for as long as one orchestrator runs.

Every box the pool holds is terminated when ``run`` returns or raises, whatever
ended it. Nothing is kept for a later launch to reattach: a box kept alive for a
resume that never comes bills for as long as nobody remembers it, so a later
launch builds its own referees and pays their setup.

Box ids are recorded in boxes.json while they are held. After a clean exit the
file names only boxes whose termination failed. After a hard kill it names every
box the dead orchestrator held, and the next ``start`` terminates those before
building anything. A box that fails at the platform level is terminated and
rebuilt from the image. Every box sits at the run's base commit for the whole
run; there is nothing to sync between rounds.
"""

from __future__ import annotations

from autoresearch.boxes.protocol import Box, BoxFactory
from autoresearch.config import RunConfig
from autoresearch.history import RunPaths, read_boxes, write_boxes
from autoresearch.referee.referee import Referee


class RefereePool:
    def __init__(self, config: RunConfig, paths: RunPaths, boxes: BoxFactory) -> None:
        self._config = config
        self._paths = paths
        self._boxes = boxes
        self._referees: dict[int, Referee] = {}
        # Boxes whose terminate raised. Kept in boxes.json and retried by terminate_all.
        self._unterminated: list[Box] = []
        self._started = False

    def _record(self) -> None:
        boxes = {f"referee:{slot}": ref.box.box_id for slot, ref in self._referees.items()}
        boxes.update({f"unterminated:{box.box_id}": box.box_id for box in self._unterminated})
        write_boxes(self._paths, boxes)

    def _terminate(self, box: Box) -> str:
        """Terminate one box. Empty on success; on failure the box is kept to retry."""
        try:
            box.terminate()
        except Exception as e:
            self._unterminated.append(box)
            return f"{box.name} ({box.box_id}): {e}"
        return ""

    def _terminate_recorded(self) -> None:
        """Terminate every box boxes.json names that is still alive: a killed orchestrator's."""
        for box_id in sorted(set(read_boxes(self._paths).values())):
            box = self._boxes.reattach(box_id)
            if box is not None:
                self._terminate(box)
        self._record()

    def _build(self, slot: int) -> Referee:
        box = self._boxes.create(name=f"referee-{self._config.run_id}-{slot}", role="referee")
        try:
            ref = Referee(box, self._config)
            ref.setup()
        except BaseException:
            # Not yet in the pool, so nothing else would ever terminate it.
            self._terminate(box)
            self._record()
            raise
        return ref

    def start(self) -> None:
        """Terminate whatever a killed orchestrator left behind, then build every slot."""
        self._terminate_recorded()
        self._started = True
        for slot in range(self._config.width):
            self._referees[slot] = self._build(slot)
            self._record()

    def get(self, slot: int) -> Referee:
        return self._referees[slot]

    def rebuild(self, slot: int) -> Referee:
        old = self._referees.pop(slot, None)
        if old is not None:
            # Already gone is fine: terminate is idempotent. Unreachable is kept to retry.
            self._terminate(old.box)
            self._record()
        ref = self._build(slot)
        self._referees[slot] = ref
        self._record()
        return ref

    def terminate_all(self) -> tuple[str, ...]:
        """Terminate every box the pool holds. A box that will not terminate never raises.

        Also retries every box whose termination failed earlier and, if ``start``
        never ran, every box boxes.json still names. Returns one line per box that
        may still be running; those stay in boxes.json for the next launch.
        """
        if not self._started:
            self._terminate_recorded()
        held = [ref.box for ref in self._referees.values()] + self._unterminated
        self._referees.clear()
        self._unterminated = []
        failures = tuple(f for box in held if (f := self._terminate(box)))
        self._record()
        return failures
