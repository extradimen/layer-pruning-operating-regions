"""Read checksum-pinned event archives for the Stage K cost lineage.

This module prepares rows only. It does not estimate an unmeasured GPU cost,
price, energy use, or allocate a shared setup event to each logical arm.
"""

import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured(event: dict, name: str):
    value = event.get("metrics", {}).get(name, {})
    if value.get("status") == "measured":
        return value["value"]
    return None


def read_cost_events(archive: Path, expected_sha256: str, stage: str):
    if sha256(archive) != expected_sha256:
        raise ValueError(f"archive SHA mismatch: {archive}")
    rows = []
    seen_events = set()
    seen_work = set()
    with tarfile.open(archive, "r:gz") as contents:
        for member in contents:
            if not member.isfile() or "/cost-" not in member.name:
                continue
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("unsafe archive member")
            if "-qwen/" in member.name:
                model = "qwen"
            elif "-mistral/" in member.name:
                model = "mistral"
            else:
                raise ValueError(f"unidentified model in {member.name}")
            source = contents.extractfile(member)
            if source is None:
                raise ValueError(f"missing data: {member.name}")
            event = json.load(source)
            event_id = event["event_id"]
            work_id = event["work_id"]
            if (stage, event_id) in seen_events or (stage, model, work_id) in seen_work:
                raise ValueError("duplicate event or work identity")
            seen_events.add((stage, event_id))
            seen_work.add((stage, model, work_id))
            wall = measured(event, "wall_s")
            cpu = measured(event, "cpu_core_s")
            if wall is not None and wall < 0 or cpu is not None and cpu < 0:
                raise ValueError("negative measured duration")
            rows.append(
                {
                    "stage": stage,
                    "model": model,
                    "run_id": event["run_id"],
                    "event_id": event_id,
                    "work_id": work_id,
                    "phase": event["phase"],
                    "scope": event["scope"],
                    "ledger": event["ledger"],
                    "point_id": event.get("point_id"),
                    "shared_consumers": event.get("shared_consumers", []),
                    "wall_s": wall,
                    "cpu_core_s": cpu,
                    "wall_status": event.get("metrics", {}).get("wall_s", {}).get("status"),
                    "cpu_status": event.get("metrics", {}).get("cpu_core_s", {}).get("status"),
                    "source_archive": str(archive),
                    "source_member": member.name,
                }
            )
    return rows
