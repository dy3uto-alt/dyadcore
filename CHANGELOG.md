# Changelog

## 0.2.0 (2026-06-15)

- **Beliefs 认知层**: 从图拓扑中涌现的认知状态，支持 auto/manual 来源
  - `add_belief()` / `get_belief()` / `get_beliefs()` — 信念 CRUD
  - `confirm_belief()` — 用户确认，置信度升至 1.0
  - `contest_belief()` — 用新证据挑战，自动合成新信念
  - `supersede_belief()` — 新旧信念取代链
  - 矛盾检测时自动合成 belief（`_synthesize_belief_from_contradiction`）
  - 纯结构置信度计算（证据量 + echoed 加成 + 源多样性 - contradicted 惩罚）
- **移除时间衰减**: 记忆不再随时间衰减，价值取决于图结构而非年龄
- **Anchor 降级**: 锚定不再参与 recall 排序，仅作为参考节点标记
- **Belief 注入**: `recall()` 新增 `include_beliefs` 参数，高置信度信念插入结果顶部
- **format_for_prompt 升级**: 分离"当前理解"（beliefs）和"关系场背景"（memories）渲染
- Contradicted 惩罚从 0.6 调整为 0.8（弱化衰减语义）
- Belief 证据加成替代 Anchor 的结构角色（`_reflection_order_factors`）

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
