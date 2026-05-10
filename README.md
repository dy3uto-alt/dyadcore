# dyadcore

**Local peer memory system for AI agents** — zero dependencies, SQLite-native, FTS5 trigram retrieval.

The user and the agent are two poles of a magnetic field, jointly leaving traces in a relational field. Memory is not data — it is a trace that *happens in the field*.

## Quick Start

```python
from dyadcore import DyadCore

dc = DyadCore("agent_memory.db")

# Write traces
uid = dc.write("用户偏好本地部署", source="user", field="技术选型")
dc.anchor(uid)  # pin important memories

# Recall with semantic search + graph expansion
results = dc.recall("本地记忆方案", field_hint="技术选型")

dc.close()
```

## Architecture: Dual Mirror

| Mirror | Source | Role |
|--------|--------|------|
| User Mirror | `source="user"` | Records user behavior and expressions |
| Agent Mirror | `source="agent"` | Records agent observations and reasoning |
| Relation Network | `reflections` table | Cross-trace edges: triggered, echoed, contradicted, related |

**Retrieval pipeline (pure SQLite):**
- **Primary**: FTS5 trigram with OR query construction (Chinese-friendly, zero-cost)
- **Fallback**: LIKE search for terms < 3 characters
- **Graph expansion**: 1-hop along reflections edges for semantic neighbors

## API Overview

### Core Operations

| Method | Description |
|--------|-------------|
| `write(content, *, source, field, memory_type)` | Write a trace, returns `memory_id` |
| `write_batch(items)` | Bulk write in a single transaction (5-10x faster) |
| `recall(query, *, field_hint, limit)` | Semantic recall with ranking |
| `anchor(memory_id)` / `unanchor(memory_id)` | Pin / unpin a memory |
| `archive(memory_id)` / `unarchive(memory_id)` | Archive / restore a memory |

### Relations

| Method | Description |
|--------|-------------|
| `add_relation(source_id, target_id, type, strength)` | Add a relation edge |
| `get_relations(memory_id)` | Get all edges for a memory |
| `get_relation_graph(memory_id, max_depth)` | Get subgraph via recursive CTE |
| `get_contradicted_map(result_ids)` | Get old→new mapping for evolution chains |

### Introspection

| Method | Description |
|--------|-------------|
| `stats()` | Total memories, active, anchored, fields, reflections |
| `list_fields()` | Field-level stats with polarity |
| `list_by_field(field)` | All memories in a field |
| `field_snapshot(field)` | Field strength, polarity, center of gravity |
| `check_silence(days)` / `check_silence_by_field(days)` | Find dormant memories |

## Design Constraints

- **Single-file SQLite** — no backend service, no config files
- **Zero external dependencies** — no sqlite-vec, no embedding model, no C extensions
- **FTS5 trigram** is the only retrieval engine (built into SQLite)
- Python >= 3.10

## Agent Integration

### Synchronous Agent Loop

```python
from dyadcore import DyadCore
from hermes_bridge import should_recall, should_write_agent_self, format_for_prompt

dc = DyadCore("agent_memory.db")
history = []
agent_self_count = 0

for turn in conversation:
    field = infer_field_with_llm(turn["user_msg"])  # use your LLM for field classification

    if should_recall(history):
        results = dc.recall(turn["user_msg"], field_hint=field, limit=5)
        memory_context = format_for_prompt(results, dyadcore=dc)
        # inject memory_context into LLM system prompt

    dc.write(turn["user_msg"], source="user", field=field)
    dc.write(turn["agent_msg"], source="agent", field=field)

    if should_write_agent_self(history, agent_self_count):
        reflection = build_agent_reflection(history)  # use your LLM
        dc.write(reflection, source="agent", field=field)
        agent_self_count += 1

    history.append({"role": "user", "content": turn["user_msg"]})
    history.append({"role": "agent", "content": turn["agent_msg"]})
```

### Async Agent Frameworks

dyadcore is synchronous (SQLite). For async agent frameworks, wrap calls with `asyncio.to_thread`:

```python
import asyncio

results = await asyncio.to_thread(dc.recall, query, field_hint=field)
dc.write(query, source="user", field=field)  # writes are fast, no need to offload
```

Or use `loop.run_in_executor` for persistent thread pool reuse.

### Bulk Import

```python
items = [
    {"content": "msg1", "source": "user", "field": "onboarding"},
    {"content": "msg2", "source": "agent", "field": "onboarding"},
    # ... hundreds more
]
ids = dc.write_batch(items)  # single transaction
```

## Running Tests

```bash
python test_dyadcore.py        # Full test suite (9 test classes)
python test_decay_topology.py  # Decay + topology verification
python test_real_data.py       # Integration test
```

## Benchmarks

```bash
python benchmark.py          # Local performance (no LLM)
python eval_retrieval.py     # Retrieval quality (MRR, Hit Rate, NDCG)
```

## License

Apache-2.0
