# PHASE 5 — TRIGGER INVESTIGATION

Execute **only Phase 5**. Do not start Phase 6.

## Objective

Investigate plausible external events that may explain each **genuine flagged event** from Phase 4.

Read:

* `Round3/data/processed/flagged_events.csv`
* `Round3/data/processed/topic_entity_by_event.csv`
* `Round3/data/processed/bluesky_enriched.csv`
* `Round3/reports/phase4_findings_notes.md`

## 1. Guard

If `flagged_events.csv` is empty, do **not** invent or search for triggers.

Create:

`Round3/reports/trigger_table.md`

Document that no trigger investigation was possible because no statistically flagged events were available, then stop.

## 2. Independent Evidence Search

Create/update:

`Round3/src/06_trigger_lookup.py`

Use an accessible news API/source such as **NewsAPI** if configured.

* Never hardcode API keys; use `Round3/.env`.
* Search using the product name plus the strongest Phase 4 topics/entities.
* For each event, search approximately **24 hours before through 6 hours after** the event.
* Record the exact search window.

Save:

`Round3/data/processed/trigger_candidates.csv`

Required fields:

* event_id
* event_timestamp
* event_type
* query
* title
* source
* published_at
* url
* description
* search_window_start
* search_window_end

Deduplicate identical URLs where possible.

## 3. Evaluate Triggers

Create:

`Round3/reports/trigger_table.md`

For **every flagged event**, include:

| What changed | When | What people discussed | Candidate explanatory event | Independent evidence | Strength/limitation |
| ------------ | ---- | --------------------- | --------------------------- | -------------------- | ------------------- |

Classify evidence only as:

* **Strong** — credible independent source, relevant, specific, and published before/during the event with consistent timing.
* **Weak** — plausible but incomplete, ambiguous, or insufficiently timed.
* **Unexplained** — no credible independent evidence or only post-event/irrelevant evidence.

Use wording such as **“consistent with”** rather than claiming causation.

Post-event articles may provide context but **cannot be presented as the cause** of an earlier event.

## 4. Validation

Check that:

* every flagged event has an entry
* sources/URLs/dates actually exist
* evidence is genuinely independent
* no trigger is fabricated
* no event is forced into Strong
* search windows and evidence timing are recorded
* limitations are stated

If the API/source is unavailable, document that limitation rather than substituting invented evidence.

Then **stop after Phase 5** and report the files created/changed and the number of events classified Strong, Weak, and Unexplained.
