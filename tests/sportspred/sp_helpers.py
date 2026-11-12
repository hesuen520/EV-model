"""Helpers shared by sportspred tests (importable without package context)."""

from __future__ import annotations


def prediction(event_id, team_ids, **over):
    rec = dict(
        created_at_utc="2030-01-01T00:00:00Z", event_id=event_id, market="winner", selection=team_ids[0],
        selection_label="Boston Celtics", probs={team_ids[0]: 0.6, team_ids[1]: 0.4}, prob=0.6,
        model_name="m", model_version="v1", validation_status="experimental", rec_status="PROVISIONAL",
    )
    rec.update(over)
    return rec
