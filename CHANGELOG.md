# Changelog

## 0.1.0 (2026-05-10)

- Initial release: renamed from MemHelix to dyadcore
- Dual Mirror architecture: user/agent peer traces with reflection graph
- FTS5 trigram retrieval with LIKE fallback for short terms
- Graph expansion: 1-hop semantic neighbor discovery
- Time decay per field with configurable half-lives
- Auto-reflection: heuristic relation building on write
- Contradiction detection: replacement/negation marker heuristics
- `write_batch()` for bulk history import
- `get_contradicted_map()` for agent-friendly evolution chain rendering
- Zero external dependencies (stdlib + SQLite only)
