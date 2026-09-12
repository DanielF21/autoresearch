"""The referee pool: one persistent referee box per worker slot for the run.

Box ids are recorded in boxes.json so a restarted orchestrator can reattach
instead of paying for new boxes. A box that fails at the platform level is
terminated and rebuilt from the image, then brought to the current incumbent.
"""

from __future__ import annotations

import contextlib

from autoresearch.boxes.protocol import BoxError, BoxFactory
from autoresearch.config import RunConfig
from autoresearch.history import RunPaths, read_boxes, write_boxes
from autoresearch.referee.referee import Referee


class RefereePool:
    def __init__(self, config: RunConfig, paths: RunPaths, boxes: BoxFactory) -> None:
        self._config = config
        self._paths = paths
        self._boxes = boxes
        self._referees: dict[int, Referee] = {}
        self._incumbent: tuple[str, str, str] | None = None  # base sha, stack diff, tree

    def _key(self, slot: int) -> str:
        return f"referee:{slot}"

    def _record(self) -> None:
        boxes = read_boxes(self._paths)
        for slot, ref in self._referees.items():
            boxes[self._key(slot)] = ref.box.box_id
        write_boxes(self._paths, boxes)

    def _build(self, slot: int) -> Referee:
        box = self._boxes.create(name=f"referee-{self._config.run_id}-{slot}", role="referee")
        ref = Referee(box, self._config)
        ref.setup()
        if self._incumbent is not None:
            ref.sync_incumbent(*self._incumbent)
        return ref

    def start(self) -> None:
        """Reattach every recorded box that is still alive; build the rest."""
        recorded = read_boxes(self._paths)
        for slot in range(self._config.width):
            box_id = recorded.get(self._key(slot))
            box = self._boxes.reattach(box_id) if box_id else None
            if box is not None:
                ref = Referee(box, self._config)
                try:
                    ref.setup()
                except BoxError:
                    box.terminate()
                    ref = self._build(slot)
            else:
                ref = self._build(slot)
            self._referees[slot] = ref
        self._record()

    def get(self, slot: int) -> Referee:
        return self._referees[slot]

    def rebuild(self, slot: int) -> Referee:
        old = self._referees.get(slot)
        if old is not None:
            with contextlib.suppress(BoxError):  # already gone is fine; that is the point
                old.box.terminate()
        ref = self._build(slot)
        self._referees[slot] = ref
        self._record()
        return ref

    def sync_all(self, base_sha: str, stack_diff: str, tree: str) -> None:
        """Bring every referee to the incumbent. A box that cannot is rebuilt once."""
        self._incumbent = (base_sha, stack_diff, tree)
        for slot in list(self._referees):
            ref = self._referees[slot]
            try:
                if ref.broken:
                    raise BoxError(ref.broken)
                ref.sync_incumbent(base_sha, stack_diff, tree)
            except BoxError:
                self.rebuild(slot)

    def terminate_all(self) -> None:
        errors: list[str] = []
        for slot, ref in self._referees.items():
            try:
                ref.box.terminate()
            except BoxError as e:
                errors.append(f"referee {slot}: {e}")
        self._referees.clear()
        write_boxes(self._paths, {})
        if errors:
            raise BoxError("; ".join(errors))
