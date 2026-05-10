# Boundary review — kaos-nlp-transformers ↔ kaos-content

**Status**: Investigation only; awaiting user signoff before any change.
**Author**: KNT-601 follow-up (audit task A4).
**Date**: 2026-05-09.

## 1. The violation

The user's stated layer cake (from the KNT-601 design discussion):

```
kaos-content (AST)  →  kaos-nlp-transformers (text)  +  kaos-nlp-core (BM25)
                            never the inverse
```

But `kaos-nlp-transformers/pyproject.toml` declares:

```toml
dependencies = [
  "kaos-core>=0.1.0a1",
  "kaos-content>=0.1.0a1",   # <-- reverse direction
  "kaos-nlp-core>=0.1.0a1",
  ...
]
```

And `kaos_nlp_transformers/clustering/semantic_dedup.py:23` imports:

```python
from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel
```

This is a real circular dependency at the package-metadata level: a fresh
`pip install kaos-content[transformers]` resolves
`kaos-content >= 0.1.0a2 → kaos-nlp-transformers >= 0.2.0a2 → kaos-content >= 0.1.0a1`.
Modern resolvers handle the cycle (it doesn't pin the same version); the
runtime call direction is one-way (kaos-nlp-transformers REGISTERS a
`SemanticDedupLevel` into kaos-content's plugin system); but the
metadata is wrong — the layer cake says one direction.

`kaos_nlp_transformers/tools.py:869` has the same import (lazy, inside
a function), so a second site to migrate.

## 2. What does kaos-nlp-transformers actually use?

Three classes from `kaos_content.dedup.types`:

| Symbol | Shape | AST coupling? |
|---|---|---|
| `DedupDocument` | `@dataclass(frozen=True, slots=True)` — fields are `doc_id: str`, `file_path: Path \| None`, `text: str \| None`, `embedding: Any \| None`, `page_images: tuple[Any, ...]`, `metadata: dict[str, Any]` | **No** — every field is a primitive or numpy-shaped. `page_images` is `tuple[Any, ...]` to keep the import lazy. |
| `DedupCluster` | frozen dataclass, only string IDs and floats | **No** |
| `DedupLevel` | `ABC` with one method, `find_clusters(documents) -> list[DedupCluster]` | **No** |

Plus there's `DedupReport` (also pure dataclass; not imported by
kaos-nlp-transformers but lives next to the types) and `kaos_content.dedup.presets`
(consumes `DedupLevel`; that's the consumer side).

Bottom line: **the types have zero AST coupling**. They only use stdlib +
numpy-shaped `Any`. They are a perfect candidate for a low-layer protocol
package — they don't belong in `kaos-content` any more than they
belong in `kaos-nlp-transformers`.

## 3. Options

### Option 1 — Status quo

Accept the cycle as a Wave-3 plugin pattern artifact.

**Pros**: zero code change.

**Cons**: violates the documented layer cake; misleading dependency
metadata; AGENTS.md in this repo says "kaos-content (AST) →
kaos-nlp-transformers (text)" but the dep arrow points the other way;
future agents can read the conflict and "fix" it in the wrong
direction.

### Option 2 — Move the protocol up to `kaos-core` (recommended long-term)

Promote `DedupDocument` / `DedupCluster` / `DedupReport` / `DedupLevel`
to a new module `kaos_core.protocols.dedup` (or `kaos_core.dedup_types`).
Both `kaos-content` and `kaos-nlp-transformers` depend on `kaos-core`
already, so this restores the layer cake immediately.

**Pros**:
- Layer cake restored cleanly.
- The types live in the package whose role is "low-layer
  protocols / shared types" — that's kaos-core's job.
- `kaos-content`'s `dedup/types.py` becomes a re-export shim
  (or just removed; kaos-content can import from kaos-core directly).
- `kaos-nlp-transformers` drops the `kaos-content` runtime dep and
  the [clustering] extra simplifies.

**Cons**:
- Coordinated release: kaos-core gets a new public module; both
  consumers bump their floor pin.
- Requires a kaos-core 0.1.0a5 (or whatever's next) release.
- Existing kaos-content consumers who import
  `from kaos_content.dedup.types import DedupDocument` need a
  re-export shim or a deprecation cycle.

### Option 3 — Move kaos-content to a `[clustering]` extra (short-term mitigation)

Today `kaos-content >= 0.1.0a1` is a BASE dep of kaos-nlp-transformers.
Move it under the existing `[clustering]` extra:

```toml
[project.optional-dependencies]
clustering = ["scipy>=1.14.1", "kaos-content>=0.2.0,<0.3"]
```

Then make `clustering/semantic_dedup.py` lazy-import the types only
when `find_clusters()` runs (currently the import is module-top-level,
which means importing `kaos_nlp_transformers` ⇒ importing
`kaos_content`).

**Pros**:
- Restores the dep-light base install promise (no kaos-content in the
  base dep tree).
- No protocol migration needed; types stay where they are.
- One-package change; ships in a single 0.2.x release.

**Cons**:
- Doesn't fix the cycle for `kaos-content[transformers]` users — that
  still pulls kaos-nlp-transformers, which under [clustering] still
  pulls kaos-content. The metadata cycle persists for that subset.
- Solves the symptom (base-install bloat) but not the underlying
  layering problem.

### Option 4 — Hybrid: ship Option 3 now, plan Option 2 for next major

1. **Now (kaos-nlp-transformers 0.2.x)**: gate `kaos-content` under
   `[clustering]` extra; lazy-import the types inside the level
   class. Ships immediately; no kaos-core release needed.
2. **Next (kaos-core 0.2.0 / kaos-content 0.2.0)**: promote the
   protocol types to `kaos-core.protocols.dedup`. Re-export shim from
   `kaos_content.dedup.types` for back-compat (deprecated; remove in
   kaos-content 0.3.0). kaos-nlp-transformers drops the
   `[clustering]` kaos-content extra; the SemanticDedupLevel imports
   from kaos-core.

**Pros**:
- Closes the immediate base-install bloat in 0.2.x.
- Schedules the principled fix for the next major cycle without
  blocking on coordinated releases.
- Each step is independently shippable.

**Cons**:
- Two coordinated changes instead of one big bang.
- The re-export shim adds a small forever-cost in kaos-content.

## 4. Recommendation

**Option 4** (hybrid).

Rationale:
- Option 1 is "do nothing", which is the wrong default for a flagged
  layer-violation.
- Option 2 alone is correct but blocks on a kaos-core release we
  don't have planned right now; coordinated cross-package work is
  expensive to schedule.
- Option 3 alone leaves the underlying problem unfixed; the cycle
  persists for `kaos-content[transformers]` users.
- Option 4 ships the immediate win in a single-package PR and plans
  the principled fix for when kaos-core's next release lands
  anyway.

## 5. Concrete steps if user approves Option 4

### 5.1 Phase A — kaos-nlp-transformers 0.2.x (single-package change)

- `pyproject.toml`: drop `kaos-content` from base `dependencies`;
  add `kaos-content>=0.1.0a2,<0.2` to the `clustering` extra.
- `kaos_nlp_transformers/clustering/semantic_dedup.py`: keep the
  module importable when kaos-content is missing (raise an
  actionable ImportError on first `find_clusters` / class
  instantiation; do NOT raise at module import).
- `kaos_nlp_transformers/tools.py:869`: same lazy-import discipline.
- Update AGENTS.md to mention the [clustering] extra requirement.
- CHANGELOG entry under [Unreleased] / next version.
- Acceptance: `pip install kaos-nlp-transformers` (no extra) does
  NOT pull `kaos-content`; `pip install kaos-nlp-transformers[clustering]`
  does. Existing test suite green.

### 5.2 Phase B — kaos-core 0.2.0 (planned, not next)

- New module `kaos_core.protocols.dedup` with the four types.
- kaos-content's `kaos_content/dedup/types.py` becomes a re-export
  shim (`from kaos_core.protocols.dedup import *` + an `__all__`
  for back-compat).
- kaos-nlp-transformers' `clustering/semantic_dedup.py` switches to
  the kaos-core import; drops the kaos-content extra entirely.

### 5.3 Phase C — kaos-content 0.3.0 (eventual cleanup)

- Drop the `kaos_content.dedup.types` re-export shim (one full minor-
  version deprecation cycle).
- The dedup pipeline orchestrator stays in kaos-content (that's
  where it belongs — it knows about `ContentDocument`); only the
  protocol types move down to kaos-core.

## 6. What would change if the user picks Option 2 directly

If the user wants to skip the interim step and move directly to
Option 2: Phase A is dropped, Phase B happens immediately. The
kaos-core release is the gating factor. Estimate ≈ half a day of work
spread across kaos-core, kaos-content, and kaos-nlp-transformers
(adding the module + shim + import switch). Risk is low because the
types are pure dataclasses with no runtime logic.

## 7. What does NOT need to change

- The runtime call direction (kaos-nlp-transformers' `SemanticDedupLevel`
  is invoked by kaos-content's pipeline) is correct as-is. We are only
  fixing the dep-graph direction, not the call direction.
- `kaos-nlp-transformers/tools.py`'s lazy import of `DedupDocument`
  inside a function body is already lazy; it only hits the dep when
  the MCP tool is invoked. That's fine after either option lands.
- kaos-content's `dedup/levels/` lexical levels (binary_hash, minhash,
  fuzzy_binary, etc.) stay in kaos-content — they are AST-aware and
  belong there.
