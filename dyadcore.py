#!/usr/bin/env python3
"""
dyadcore v0.1.0 — 本地 Peer 记忆系统

用户与 Agent 是磁场的两极，共同在关系场（Field）中留下痕迹。
记忆不是数据，而是"发生在场中的痕迹"。

Dual Mirror 架构：
  - 用户镜：source='user' 的痕迹，记录用户的行为和表达
  - Agent镜：source='agent' 的痕迹，记录 Agent 的观察和推理
  - 关系网：reflections 表存储跨轨迹的边，在召回时做图扩展

检索策略（纯 SQLite）：
  - 主引擎：FTS5 trigram + OR 查询构建，零成本，中文友好
  - 回退引擎：LIKE 搜索（<3 字符短词，trigram 无法覆盖）
  - 图扩展：沿 reflections 边发现语义邻居（1-hop）

核心约束：
  - 单文件 SQLite，零后台服务，零配置文件
  - 零外部依赖：无 sqlite-vec，无 embedding 模型，无 C 扩展
  - FTS5 trigram 为唯一检索引擎（SQLite 内置）
"""

import json
import re
import sqlite3
import time
import os
from typing import Optional, Any


class DyadCore:
    """本地 Peer 记忆系统 — Dual Mirror 架构。

    用法：
        dc = DyadCore("dyadcore.db")
        mid = dc.write("用户偏好本地部署", source="user", field="技术选型")
        dc.anchor(mid)
        results = dc.recall("本地记忆", field_hint="技术选型")
        dc.close()
    """

    def __init__(self, db_path: str = "dyadcore.db",
                 contradicted_enabled: bool = True, echoed_enabled: bool = True):
        self.db_path = db_path
        self.contradicted_enabled = contradicted_enabled
        self.echoed_enabled = echoed_enabled
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Backward-compatible stubs (no-op in Dual Mirror)
    # ------------------------------------------------------------------

    @property
    def embedding_tier(self) -> int:
        """始终返回 3 (FTS5-only)。Dual Mirror 无 embedding 依赖。"""
        return 3

    @property
    def embedding_dim(self) -> None:
        """始终返回 None。Dual Mirror 无 embedding 依赖。"""
        return None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """初始化数据库表（幂等）。

        Dual Mirror 架构：
          - memories: 用户和 Agent 的对等痕迹表
          - memory_fts: FTS5 trigram 全文索引
          - reflections: 跨轨迹关系边
          - beliefs: 从图拓扑中涌现的认知状态（闭环断点 ③）
        """
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT CHECK(memory_type IN ('utterance', 'action', 'meta')),
                source TEXT CHECK(source IN ('user', 'agent')),
                field TEXT,
                anchor INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                accessed_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content,
                tokenize='trigram',
                content='memories',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS mem_fts_insert AFTER INSERT ON memories BEGIN
                INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS mem_fts_delete AFTER DELETE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS mem_fts_update AFTER UPDATE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT CHECK(relation_type IN ('triggered', 'echoed', 'contradicted', 'related')),
                strength REAL DEFAULT 0.5,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (source_id) REFERENCES memories(id),
                FOREIGN KEY (target_id) REFERENCES memories(id)
            );

            CREATE INDEX IF NOT EXISTS idx_reflections_source ON reflections(source_id);
            CREATE INDEX IF NOT EXISTS idx_reflections_target ON reflections(target_id);
            CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(relation_type);

            CREATE INDEX IF NOT EXISTS idx_memories_field ON memories(field);
            CREATE INDEX IF NOT EXISTS idx_memories_anchor ON memories(anchor);
            CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
            CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);

            CREATE TABLE IF NOT EXISTS beliefs (
                id INTEGER PRIMARY KEY,
                subject TEXT NOT NULL,
                statement TEXT NOT NULL,
                field TEXT,
                confidence REAL DEFAULT 0.5,
                evidence_ids TEXT NOT NULL DEFAULT '[]',
                contradicted_by TEXT DEFAULT '[]',
                verification_status TEXT DEFAULT 'unverified'
                    CHECK(verification_status IN ('confirmed', 'contested', 'unverified', 'stale')),
                source TEXT DEFAULT 'auto'
                    CHECK(source IN ('auto', 'manual')),
                superseded_by INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_beliefs_field ON beliefs(field);
            CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON beliefs(subject);
            CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(verification_status);
        """)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> float:
        return time.time()

    def _touch(self, memory_id: int) -> None:
        self.conn.execute(
            "UPDATE memories SET accessed_at = ? WHERE id = ?",
            (self._now(), memory_id)
        )

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return dict(row) if row else None

    def _rows_to_dicts(self, rows) -> list[dict]:
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def write(
        self,
        content: str,
        memory_type: str = "utterance",
        source: str = "user",
        field: Optional[str] = None,
    ) -> int:
        """写入一条痕迹。

        Args:
            content: 痕迹内容。
            memory_type: utterance | action | meta
            source: user | agent — 对等实体，无特殊类别
            field: 关系场名称（可选）。

        Returns:
            新痕迹的 memory_id。
        """
        now = self._now()
        cur = self.conn.execute(
            "INSERT INTO memories (content, memory_type, source, field, created_at, accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, memory_type, source, field, now, now)
        )
        memory_id = cur.lastrowid

        # 启发式自动建边
        self._build_reflections(memory_id, content, field, source)

        return memory_id

    def write_batch(self, items: list[dict]) -> list[int]:
        """批量写入痕迹（单事务，比逐条 write() 快 5-10x）。

        Args:
            items: [{"content": ..., "memory_type": "utterance", "source": "user", "field": ...}, ...]
                content 必填，其余字段可选，默认值同 write()

        Returns:
            [memory_id, ...]
        """
        ids = []
        now = self._now()
        with self.conn:
            for item in items:
                cur = self.conn.execute(
                    "INSERT INTO memories (content, memory_type, source, field, created_at, accessed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (item["content"],
                     item.get("memory_type", "utterance"),
                     item.get("source", "user"),
                     item.get("field"),
                     now, now)
                )
                mid = cur.lastrowid
                ids.append(mid)
                self._build_reflections(mid, item["content"], item.get("field"), item.get("source", "user"))
        return ids

    def anchor(self, memory_id: int) -> None:
        self.conn.execute(
            "UPDATE memories SET anchor = 1, accessed_at = ? WHERE id = ?",
            (self._now(), memory_id)
        )

    def unanchor(self, memory_id: int) -> None:
        self.conn.execute(
            "UPDATE memories SET anchor = 0, accessed_at = ? WHERE id = ?",
            (self._now(), memory_id)
        )

    def archive(self, memory_id: int) -> None:
        self.conn.execute(
            "UPDATE memories SET archived = 1, accessed_at = ? WHERE id = ?",
            (self._now(), memory_id)
        )

    def unarchive(self, memory_id: int) -> None:
        self.conn.execute(
            "UPDATE memories SET archived = 0, accessed_at = ? WHERE id = ?",
            (self._now(), memory_id)
        )

    def set_field(self, memory_id: int, field: Optional[str]) -> None:
        self.conn.execute(
            "UPDATE memories SET field = ?, accessed_at = ? WHERE id = ?",
            (field, self._now(), memory_id)
        )

    # ------------------------------------------------------------------
    # Beliefs — 从图拓扑涌现的认知状态
    # ------------------------------------------------------------------

    def get_beliefs(
        self, field: Optional[str] = None, subject: Optional[str] = None,
        status: Optional[str] = None, include_superseded: bool = False
    ) -> list[dict]:
        """查询 beliefs，可按 field/subject/status 过滤。

        返回按 confidence DESC 排序的信念列表。
        """
        conditions = []
        params: list = []
        if not include_superseded:
            conditions.append("superseded_by = 0")
        if field is not None:
            conditions.append("field = ?")
            params.append(field)
        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if status is not None:
            conditions.append("verification_status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"SELECT * FROM beliefs {where} ORDER BY confidence DESC",
            params
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_belief(self, belief_id: int) -> Optional[dict]:
        """获取单个 belief。"""
        row = self.conn.execute(
            "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def add_belief(
        self, subject: str, statement: str, field: str,
        evidence_ids: Optional[list[int]] = None,
        confidence: float = 0.5
    ) -> int:
        """手动声明一个 belief（Agent 或用户显式表达对用户的理解）。"""
        now = self._now()
        ev_ids = json.dumps(evidence_ids or [])
        cursor = self.conn.execute(
            "INSERT INTO beliefs (subject, statement, field, confidence, "
            "evidence_ids, verification_status, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'unverified', 'manual', ?, ?)",
            (subject, statement, field, confidence, ev_ids, now, now)
        )
        return cursor.lastrowid

    def confirm_belief(self, belief_id: int) -> None:
        """用户确认一个 belief——置信度设为 1.0，状态设为 confirmed。"""
        self.conn.execute(
            "UPDATE beliefs SET verification_status = 'confirmed', "
            "confidence = 1.0, updated_at = ? WHERE id = ?",
            (self._now(), belief_id)
        )

    def contest_belief(self, belief_id: int, contradicting_memory_id: int) -> Optional[int]:
        """用新证据挑战一个现有 belief。

        自动创建新 belief（从 contradicting memory 合成），
        旧 belief 标记为 stale。
        返回新 belief ID，或 None（如果旧 belief 不存在）。
        """
        old_belief = self.get_belief(belief_id)
        if not old_belief:
            return None

        new_mem = self.conn.execute(
            "SELECT content, field FROM memories WHERE id = ?",
            (contradicting_memory_id,)
        ).fetchone()
        if not new_mem:
            return None

        # 标记旧信念为 stale
        now = self._now()
        self.conn.execute(
            "UPDATE beliefs SET verification_status = 'stale', "
            "updated_at = ? WHERE id = ?",
            (now, belief_id)
        )

        # 合并证据链
        old_evidence = json.loads(old_belief["evidence_ids"])
        old_contra = json.loads(old_belief.get("contradicted_by") or "[]")
        new_evidence = list(set(old_evidence + [contradicting_memory_id]))
        new_contra = list(set(old_contra + old_evidence))

        confidence = self._compute_confidence(new_evidence, new_contra)
        return self.add_belief(
            subject=old_belief["subject"],
            statement=new_mem["content"],
            field=new_mem["field"] or old_belief["field"],
            evidence_ids=new_evidence,
            confidence=confidence
        )

    def update_belief_confidence(self, belief_id: int) -> float:
        """根据当前证据图状态重新计算 confidence。返回新值。"""
        row = self.conn.execute(
            "SELECT evidence_ids, contradicted_by FROM beliefs WHERE id = ?",
            (belief_id,)
        ).fetchone()
        if not row:
            return 0.0

        evidence_ids = json.loads(row["evidence_ids"])
        contradicted_ids = json.loads(row["contradicted_by"] or "[]")
        confidence = self._compute_confidence(evidence_ids, contradicted_ids)
        self.conn.execute(
            "UPDATE beliefs SET confidence = ?, updated_at = ? WHERE id = ?",
            (confidence, self._now(), belief_id)
        )
        return confidence

    def supersede_belief(self, old_id: int, new_id: int) -> None:
        """标记旧信念被新信念取代。"""
        now = self._now()
        self.conn.execute(
            "UPDATE beliefs SET verification_status = 'stale', "
            "superseded_by = ?, updated_at = ? WHERE id = ?",
            (new_id, now, old_id)
        )

    # ------------------------------------------------------------------
    # Reflections — 跨轨迹关系网
    # ------------------------------------------------------------------

    REFLECTION_WINDOW_SEC = 3600       # 1 小时内写入的记忆参与自动建边
    REFLECTION_OVERLAP_THRESHOLD = 1   # 关键词重叠 >= 此值触发 related

    # 记忆不随时间衰减——旧记忆是未点燃的信号，价值取决于图结构而非年龄
    FIELD_BONUS = 1.5     # 同 field 匹配乘子（bucket 软加权）

    # Tier 1 闭环：reflections 层信号回流召回排序
    CONTRADICTED_PENALTY = 0.8   # 被 contradicted 指向的旧记忆微降权（弱化，主信号在拓扑渲染）
    ECHOED_BONUS = 1.1           # 有 echoed 边的记忆微升权（确认信号）
    FIELD_STRENGTH_CAP = 1.15    # 场强加成上限
    FIELD_STRENGTH_SCALE = 0.05  # LN(count+1) × scale

    # Tier 2 完善：自动 contradicted 检测
    CONTRADICTED_SCAN_LIMIT = 30  # 写入时扫描同 field 最近 N 条（不限时间窗口）
    CONTRADICTED_AUTO_STRENGTH = 0.9  # 自动建边的强度

    # 替换/否定标记词 — 显式修正信号
    CONTRADICTED_MARKERS_ZH = [
        "不再", "换成", "改为", "改用", "换成了", "不用了",
        "已切换到", "切换到了", "换到", "换了", "不再用",
        "替代了", "取代了", "换掉", "换掉了", "弃用",
        "已迁移到", "迁移到了", "淘汰了",
    ]
    CONTRADICTED_MARKERS_EN = [
        "switched to", "switched from", "replaced", "dropped",
        "moved from", "instead of", "no longer",
        "migrated to", "migrated from", "upgraded to",
    ]

    def _build_reflections(
        self, new_id: int, content: str, field: Optional[str], source: str
    ) -> None:
        """启发式自动建边：新记忆与同场近期记忆的关系发现。

        对话级连接（1h 窗口）：
          - 关键词重叠 >= threshold → related
          - related + 不同 source + 高相似度 → echoed
          - 高重叠 → triggered

        知识级连接（不限窗口）：
          - 替换/否定标记词 + 共享实体 → contradicted (Tier 2)
        """
        if field is None:
            return

        now = self._now()
        cutoff = now - self.REFLECTION_WINDOW_SEC

        # --- 对话级：1h 窗口内 related / echoed / triggered ---
        candidates = self.conn.execute(
            "SELECT id, content, source FROM memories "
            "WHERE field = ? AND id != ? AND created_at >= ? AND archived = 0",
            (field, new_id, cutoff)
        ).fetchall()

        if candidates:
            new_tokens = self._tokenize(content)
            if new_tokens:
                for c in candidates:
                    c_tokens = self._tokenize(c["content"])
                    if not c_tokens:
                        continue

                    overlap = len(new_tokens & c_tokens)
                    if overlap < self.REFLECTION_OVERLAP_THRESHOLD:
                        continue

                    union = len(new_tokens | c_tokens)
                    strength = round(overlap / union, 3) if union > 0 else 0.5

                    if c["source"] != source and strength >= 0.4:
                        rel_type = "echoed"
                    elif overlap >= 4:
                        rel_type = "triggered"
                    else:
                        rel_type = "related"

                    self.conn.execute(
                        "INSERT OR IGNORE INTO reflections "
                        "(source_id, target_id, relation_type, strength, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (new_id, c["id"], rel_type, strength, now)
                    )
                    self.conn.execute(
                        "INSERT OR IGNORE INTO reflections "
                        "(source_id, target_id, relation_type, strength, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (c["id"], new_id, rel_type, strength, now)
                    )

        # --- 知识级：跨窗口自动 contradicted 检测 (Tier 2) ---
        if self.contradicted_enabled:
            self._build_contradicted_reflections(new_id, content, field)

    def _build_contradicted_reflections(
        self, new_id: int, content: str, field: str
    ) -> None:
        """跨窗口扫描同 field 最近 N 条记忆，检测矛盾并自动建 contradicted 边。

        与 _build_reflections 不同：无时间窗口限制，仅检测 contradicted。
        被锚定的记忆优先进入候选池——anchor 作为参考节点标记，不参与 ranking。
        """
        candidates = self.conn.execute(
            "SELECT id, content FROM memories "
            "WHERE field = ? AND id != ? AND archived = 0 "
            "ORDER BY anchor DESC, created_at DESC LIMIT ?",
            (field, new_id, self.CONTRADICTED_SCAN_LIMIT)
        ).fetchall()

        if not candidates:
            return

        for c in candidates:
            # 跳过已有 contradicted 边的配对（避免重复建边）
            existing = self.conn.execute(
                "SELECT 1 FROM reflections WHERE relation_type = 'contradicted' "
                "AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))",
                (new_id, c["id"], c["id"], new_id)
            ).fetchone()
            if existing:
                continue

            if self._detect_contradiction(content, c["content"]):
                self.add_relation(
                    new_id, c["id"], "contradicted",
                    strength=self.CONTRADICTED_AUTO_STRENGTH
                )
                self._synthesize_belief_from_contradiction(
                    new_id, c["id"], field
                )

    def _synthesize_belief_from_contradiction(
        self, winning_id: int, losing_id: int, field: str
    ) -> Optional[int]:
        """当检测到矛盾时，从获胜记忆自动合成 belief。

        winning_id: 新记忆（矛盾声明的发起者）
        losing_id: 旧记忆（被矛盾指向的）
        返回新 belief 的 ID，或 None（如果该 subject 已有 confirmed 信念）。
        """
        # 检查是否已有同一 subject 的 confirmed 信念
        subject = self._extract_subject(winning_id)
        existing = self.conn.execute(
            "SELECT id, verification_status FROM beliefs "
            "WHERE subject = ? AND field = ? AND superseded_by = 0 "
            "ORDER BY created_at DESC LIMIT 1",
            (subject, field)
        ).fetchone()

        if existing and existing["verification_status"] == "confirmed":
            # 已确认的信念不会被自动覆盖——需要显式 contest
            return None

        # 计算初始置信度
        evidence_ids = json.dumps([winning_id])
        contradicted_by = json.dumps([losing_id])
        confidence = self._compute_confidence([winning_id], [losing_id])

        now = self._now()
        if existing:
            # 取代旧信念
            self.conn.execute(
                "UPDATE beliefs SET superseded_by = ?, updated_at = ? WHERE id = ?",
                (0, now, existing["id"])  # 先占位，下面更新
            )
            # 实际写入后在下面更新 superseded_by
            self.conn.execute(
                "UPDATE beliefs SET verification_status = 'stale', "
                "superseded_by = -1, updated_at = ? WHERE id = ?",
                (now, existing["id"])
            )

        cursor = self.conn.execute(
            "INSERT INTO beliefs (subject, statement, field, confidence, "
            "evidence_ids, contradicted_by, verification_status, source, "
            "created_at, updated_at) "
            "SELECT ?, content, ?, ?, ?, ?, 'unverified', 'auto', ?, ? "
            "FROM memories WHERE id = ?",
            (subject, field, confidence, evidence_ids, contradicted_by, now, now, winning_id)
        )
        new_belief_id = cursor.lastrowid

        if existing:
            self.conn.execute(
                "UPDATE beliefs SET superseded_by = ? WHERE id = ?",
                (new_belief_id, existing["id"])
            )

        return new_belief_id

    def _extract_subject(self, memory_id: int) -> str:
        """从记忆内容中提取 belief subject 键。

        策略：取 field 作为前缀 + 前 3 个关键 token 的 hash。
        足够区分同一 field 内的不同主题。
        """
        row = self.conn.execute(
            "SELECT field, content FROM memories WHERE id = ?",
            (memory_id,)
        ).fetchone()
        if not row:
            return f"unknown_{memory_id}"

        field = row["field"] or "general"
        content = row["content"]

        # 简单提取：取前 60 个字符作为 subject 摘要
        snippet = content[:60].replace("\n", " ").strip()
        return f"{field}:{snippet}"

    def _compute_confidence(
        self, evidence_ids: list[int], contradicted_ids: list[int]
    ) -> float:
        """纯结构置信度计算——从图拓扑中涌现，无外部权威。

        四个因子：
          1. 证据数量（base）
          2. Echoed 边加成（跨源确认信号）
          3. 源多样性加成（user + agent 双源 > 单源）
          4. Contradicted 惩罚
        """
        n_evidence = max(len(evidence_ids), 1)
        base = min(1.0, 0.5 + n_evidence * 0.1)  # 1→0.6, 2→0.7, 3→0.8...

        # Echoed bonus
        if evidence_ids:
            ph = ','.join('?' for _ in evidence_ids)
            echoed_count = self.conn.execute(
                f"SELECT COUNT(DISTINCT id) FROM reflections "
                f"WHERE relation_type = 'echoed' "
                f"AND (source_id IN ({ph}) OR target_id IN ({ph}))",
                evidence_ids + evidence_ids
            ).fetchone()[0]
            echo_bonus = min(0.2, echoed_count * 0.05)
        else:
            echo_bonus = 0.0

        # Source diversity
        if evidence_ids:
            ph = ','.join('?' for _ in evidence_ids)
            sources = self.conn.execute(
                f"SELECT COUNT(DISTINCT source) FROM memories WHERE id IN ({ph})",
                evidence_ids
            ).fetchone()[0]
            diversity_bonus = 0.1 if sources >= 2 else 0.0
        else:
            diversity_bonus = 0.0

        # Contradicted penalty
        contra_penalty = len(contradicted_ids) * 0.15

        return round(max(0.1, min(1.0, base + echo_bonus + diversity_bonus - contra_penalty)), 4)

    def _detect_contradiction(self, new_content: str, old_content: str) -> bool:
        """检测新记忆是否构成对旧记忆的修正/否定。

        两个条件同时满足才判定为矛盾：
          1. 新记忆包含替换/否定标记词（中文或英文）
          2. 新旧记忆共享至少一个关键实体（同一主题）
        """
        # 条件 1：替换/否定标记词
        has_marker = False
        for m in self.CONTRADICTED_MARKERS_ZH:
            if m in new_content:
                has_marker = True
                break
        if not has_marker:
            new_lower = new_content.lower()
            for m in self.CONTRADICTED_MARKERS_EN:
                if m in new_lower:
                    has_marker = True
                    break
        if not has_marker:
            return False

        # 条件 2：共享关键实体（排除标记词 — 标记词是矛盾信号而非主题信号）
        marker_stop = set()
        for m in self.CONTRADICTED_MARKERS_ZH + self.CONTRADICTED_MARKERS_EN:
            # 标记词可能作为 bigram 出现在实体集中，需要排除
            marker_stop.add(m.lower())
            # 同时排除标记词的子串 bigram（如"不再"来自"不再使用"）
            for i in range(len(m) - 1):
                if m[i:i+2]:
                    marker_stop.add(m[i:i+2].lower())

        new_entities = self._extract_entities(new_content) - marker_stop
        old_entities = self._extract_entities(old_content) - marker_stop
        if not new_entities or not old_entities:
            return False

        # >= 2 共享实体才判定同主题 — 过滤后的高置信度匹配
        return len(new_entities & old_entities) >= 2

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """从文本中提取关键实体（>=3字中文词 或 >=4字母英文词）。

        用于矛盾检测中的主题匹配 — 两个记忆共享实体 = 同一主题。
        """
        cleaned = re.sub(r"[^\u4e00-\u9fff\w]", " ", text)
        entities: set[str] = set()
        for seg in cleaned.split():
            seg = seg.strip().lower()
            if len(seg) >= 3:
                entities.add(seg)
                # CJK bigram：中文偏旁级部分匹配
                if any('\u4e00' <= ch <= '\u9fff' for ch in seg):
                    for i in range(len(seg) - 1):
                        entities.add(seg[i:i + 2])
        return entities

    def add_relation(
        self, source_id: int, target_id: int, relation_type: str, strength: float = 0.5
    ) -> int:
        """显式添加关系边。

        Args:
            source_id: 源记忆 ID。
            target_id: 目标记忆 ID。
            relation_type: triggered | echoed | contradicted | related
            strength: 关系强度 0.0-1.0。

        Returns:
            新 relation 的 ID。
        """
        existing = self.conn.execute(
            "SELECT id FROM reflections "
            "WHERE source_id = ? AND target_id = ? AND relation_type = ? "
            "LIMIT 1",
            (source_id, target_id, relation_type)
        ).fetchone()

        if existing:
            self.conn.execute(
                "UPDATE reflections SET strength = ? WHERE id = ?",
                (strength, existing["id"])
            )
            return existing["id"]

        cur = self.conn.execute(
            "INSERT INTO reflections (source_id, target_id, relation_type, strength, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, relation_type, strength, self._now())
        )
        return cur.lastrowid

    def get_relations(self, memory_id: int) -> list[dict]:
        """获取某记忆的所有关系边。"""
        rows = self.conn.execute(
            "SELECT r.*, "
            "m_s.content AS source_content, m_t.content AS target_content "
            "FROM reflections r "
            "JOIN memories m_s ON r.source_id = m_s.id "
            "JOIN memories m_t ON r.target_id = m_t.id "
            "WHERE r.source_id = ? OR r.target_id = ?",
            (memory_id, memory_id)
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_relation_graph(self, memory_id: int, max_depth: int = 2) -> list[dict]:
        """获取某记忆的邻域关系子图（recursive CTE）。

        Returns:
            包含 rel_id, source_id, target_id, relation_type, strength,
            depth, source_content, target_content 的边列表。
        """
        sql = """
            WITH RECURSIVE graph AS (
                SELECT r.id AS rel_id, r.source_id, r.target_id,
                       r.relation_type, r.strength, 0 AS depth
                FROM reflections r
                WHERE r.source_id = ? OR r.target_id = ?

                UNION ALL

                SELECT r.id, r.source_id, r.target_id,
                       r.relation_type, r.strength * 0.5, g.depth + 1
                FROM reflections r
                JOIN graph g ON (r.source_id IN (g.source_id, g.target_id)
                              OR r.target_id IN (g.source_id, g.target_id))
                WHERE g.depth < ?
                  AND r.id != g.rel_id
            )
            SELECT DISTINCT g.*,
                m_s.content AS source_content, m_t.content AS target_content
            FROM graph g
            JOIN memories m_s ON g.source_id = m_s.id
            JOIN memories m_t ON g.target_id = m_t.id
            ORDER BY g.depth, g.strength DESC
        """
        rows = self.conn.execute(sql, (memory_id, memory_id, max_depth)).fetchall()
        return self._rows_to_dicts(rows)

    def get_contradicted_map(self, result_ids: list[int]) -> dict[int, int]:
        """返回 old_id -> new_id 的 contradicted 演化映射。

        用于 format_for_prompt 渲染演化链，不再需要外部代码直接访问 .conn。

        Args:
            result_ids: recall() 返回结果中的 id 列表

        Returns:
            {old_memory_id: new_memory_id, ...} 的字典
        """
        if len(result_ids) < 2:
            return {}
        ph = ','.join('?' for _ in result_ids)
        edges = self.conn.execute(
            f"SELECT source_id, target_id FROM reflections "
            f"WHERE relation_type = 'contradicted' "
            f"AND source_id IN ({ph}) AND target_id IN ({ph})",
            result_ids + result_ids
        ).fetchall()
        contradicted_map = {}
        for src_id, tgt_id in edges:
            src = self.get_memory(src_id)
            tgt = self.get_memory(tgt_id)
            if src and tgt:
                if src.get('created_at', 0) >= tgt.get('created_at', 0):
                    contradicted_map[tgt_id] = src_id
                else:
                    contradicted_map[src_id] = tgt_id
        return contradicted_map

    # ------------------------------------------------------------------
    # Recall API
    # ------------------------------------------------------------------

    def recall(
        self,
        query: Optional[str] = None,
        field_hint: Optional[str] = None,
        limit: int = 5,
        include_archived: bool = False,
        expand_graph: bool = True,
        include_beliefs: bool = True,
    ) -> list[dict]:
        """召回记忆。纯 FTS5 + LIKE + 关系图扩展，零外部依赖。

        检索策略：
          主引擎 — FTS5 trigram（零成本，始终可用）
          回退引擎 — LIKE 搜索（trigram 无法覆盖的 <3 字符短词）
          图扩展 — 沿 reflections 边发现语义邻居（1-hop）
          Belief 注入 — 当前 field 的高置信度信念插入结果顶部

        Args:
            query: 可选搜索词。
            field_hint: 当前关系场名称。
            limit: 返回最大条数。
            include_archived: 是否包含已归档记忆。
            expand_graph: 是否启用关系图扩展。
            include_beliefs: 是否在结果中注入匹配的 beliefs。

        Returns:
            记忆 dict 列表，按相关性从高到低排列。
            每条 dict 含 memories 表全部字段，外加：
              - rank (float, 可选): FTS5 BM25 分数（越小越好）
              - snippet (str, 可选): 命中片段
              - depth (int): 0=直接命中, 1=图扩展命中, 2=belief
              - via_type (str, 可选): 关系类型
              - is_belief (bool, 可选): True 表示这是信念而非记忆
        """
        if query:
            has_short = self._has_short_terms(query)
            fts_limit = max(1, limit - 1) if has_short else limit

            # 主引擎：FTS5 trigram
            results = self._recall_fts(query, field_hint, fts_limit, include_archived)

            # 回退引擎：LIKE 搜索
            if has_short:
                like_results = self._recall_like(query, field_hint, limit, include_archived)
                results = self._merge_results(results, like_results, limit)

            # 关系图扩展
            if expand_graph and len(results) < limit:
                graph_results = self._recall_graph(results, limit - len(results), include_archived)
                results = self._merge_results(results, graph_results, limit)

            # Belief 注入：高置信度信念优先于原始记忆
            if include_beliefs and field_hint:
                belief_results = self._recall_beliefs(field_hint, max(1, limit // 3))
                if belief_results:
                    results = belief_results + results
                    if len(results) > limit:
                        results = results[:limit]

            return results
        else:
            results = self._recall_no_query(field_hint, limit, include_archived)
            # 无查询时也注入 beliefs
            if include_beliefs and field_hint:
                belief_results = self._recall_beliefs(field_hint, max(1, limit // 3))
                if belief_results:
                    results = belief_results + results
                    if len(results) > limit:
                        results = results[:limit]
            return results

    def _recall_fts(
        self, query: str, field_hint: Optional[str],
        limit: int, include_archived: bool
    ) -> list[dict]:
        """FTS5 全文检索。"""
        fts_query = self._build_fts5_query(query)
        if not fts_query:
            return self._recall_no_query(field_hint, limit, include_archived)

        archived_filter = "" if include_archived else "AND m.archived = 0"

        if field_hint is not None:
            sql = f"""
                WITH fts_results AS (
                    SELECT rowid AS memory_id, bm25(memory_fts, 1.0, 0.75) AS rank,
                           snippet(memory_fts, 0, '<mark>', '</mark>', '...', 32) AS snippet
                    FROM memory_fts
                    WHERE content MATCH ?
                    LIMIT 100
                )
                SELECT
                    m.*,
                    f.rank,
                    f.snippet,
                    0 AS depth
                FROM memories m
                JOIN fts_results f ON m.id = f.memory_id
                WHERE 1=1 {archived_filter}
                ORDER BY
                    (-COALESCE(f.rank, 1.0))
                    * CASE WHEN m.field = ?2 THEN {self.FIELD_BONUS} ELSE 1.0 END
                    {self._reflection_order_factors()} DESC
                LIMIT ?3
            """
            rows = self.conn.execute(sql, (fts_query, field_hint, limit)).fetchall()
        else:
            sql = f"""
                WITH fts_results AS (
                    SELECT rowid AS memory_id, bm25(memory_fts, 1.0, 0.75) AS rank,
                           snippet(memory_fts, 0, '<mark>', '</mark>', '...', 32) AS snippet
                    FROM memory_fts
                    WHERE content MATCH ?
                    LIMIT 100
                )
                SELECT
                    m.*,
                    f.rank,
                    f.snippet,
                    0 AS depth
                FROM memories m
                JOIN fts_results f ON m.id = f.memory_id
                WHERE 1=1 {archived_filter}
                ORDER BY
                    (-COALESCE(f.rank, 1.0))
                    {self._reflection_order_factors()} DESC
                LIMIT ?2
            """
            rows = self.conn.execute(sql, (fts_query, limit)).fetchall()

        return self._rows_to_dicts(rows)

    def _recall_like(
        self, query: str, field_hint: Optional[str],
        limit: int, include_archived: bool
    ) -> list[dict]:
        """LIKE 回退：匹配 trigram 无法索引的短词（<3 字符）。

        FTS5 trigram tokenizer 丢弃所有 <3 字符的 token，导致
        单字 CJK（如"猫"）、双字 CJK（如"异步"）、短缩写（如"CI"）
        无法通过 FTS5 匹配。此方法用 LIKE '%term%' 兜底。
        """
        cleaned = ''.join(
            ch if ch.isalnum() or ch.isspace() else ' '
            for ch in query
        ).strip()
        tokens = cleaned.split()
        short_terms = [t for t in tokens if len(t) < 3]

        if not short_terms:
            return []

        archived_filter = "" if include_archived else "AND archived = 0"
        like_clause = ' OR '.join(['content LIKE ?' for _ in short_terms])

        if field_hint is not None:
            sql = f"""
                SELECT m.*, NULL AS rank, NULL AS snippet, 0 AS depth
                FROM memories m
                WHERE ({like_clause})
                    {archived_filter}
                ORDER BY
                    CASE WHEN m.field = ? THEN {self.FIELD_BONUS} ELSE 1.0 END
                    {self._reflection_order_factors()} DESC
                LIMIT ?
            """
            params = [field_hint]
            params.extend([f'%{t}%' for t in short_terms])
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()
        else:
            sql = f"""
                SELECT m.*, NULL AS rank, NULL AS snippet, 0 AS depth
                FROM memories m
                WHERE ({like_clause})
                    {archived_filter}
                ORDER BY
                    1.0
                    {self._reflection_order_factors()} DESC
                LIMIT ?
            """
            params = [f'%{t}%' for t in short_terms]
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()

        return self._rows_to_dicts(rows)

    def _recall_graph(
        self, seeds: list[dict], limit: int, include_archived: bool
    ) -> list[dict]:
        """关系图扩展：从 FTS5 命中沿 reflections 边发现邻居（1-hop）。"""
        if not seeds:
            return []

        seed_ids = [s['id'] for s in seeds]
        ph = ','.join('?' for _ in seed_ids)
        archived_filter = "" if include_archived else "AND m.archived = 0"

        # 1-hop 扩展：找出与种子通过 reflections 关联的记忆
        # source_id → target_id 方向：种子是 source，邻居是 target
        # target_id → source_id 方向：种子是 target，邻居是 source
        sql = f"""
            SELECT DISTINCT m.*,
                r.strength AS graph_strength,
                1 AS depth,
                r.relation_type AS via_type
            FROM reflections r
            JOIN memories m ON (
                (r.source_id IN ({ph}) AND m.id = r.target_id)
                OR
                (r.target_id IN ({ph}) AND m.id = r.source_id)
            )
            WHERE m.id NOT IN ({ph})
              {archived_filter}
            ORDER BY
                r.strength
                {self._reflection_order_factors()} DESC
            LIMIT ?
        """
        # params: seeds for source_id IN, seeds for target_id IN, seeds for NOT IN, limit
        params = list(seed_ids) + list(seed_ids) + list(seed_ids) + [limit]
        rows = self.conn.execute(sql, params).fetchall()
        return self._rows_to_dicts(rows)

    def _recall_beliefs(self, field: str, limit: int) -> list[dict]:
        """检索匹配 field 的高置信度 beliefs。

        返回格式与 memories 兼容的 dict 列表，
        添加 is_belief=True, depth=2 标记供下游区分。
        """
        rows = self.conn.execute(
            "SELECT * FROM beliefs "
            "WHERE field = ? AND superseded_by = 0 AND confidence >= 0.5 "
            "ORDER BY confidence DESC LIMIT ?",
            (field, limit)
        ).fetchall()

        results = []
        for r in rows:
            d = self._row_to_dict(r)
            d["is_belief"] = True
            d["depth"] = 2
            d["source"] = "belief"
            d["content"] = d["statement"]
            results.append(d)
        return results

    def _recall_no_query(
        self, field_hint: Optional[str], limit: int, include_archived: bool
    ) -> list[dict]:
        """无查询词时的召回：同场优先 × 图拓扑排序。"""
        archived_filter = "" if include_archived else "AND archived = 0"

        if field_hint is not None:
            sql = f"""
                SELECT m.*, 0 AS depth
                FROM memories m
                WHERE 1=1 {archived_filter}
                ORDER BY
                    CASE WHEN m.field = ?1 THEN {self.FIELD_BONUS} ELSE 1.0 END
                    {self._reflection_order_factors()} DESC
                LIMIT ?2
            """
            rows = self.conn.execute(sql, (field_hint, limit)).fetchall()
        else:
            sql = f"""
                SELECT m.*, 0 AS depth
                FROM memories m
                WHERE 1=1 {archived_filter}
                ORDER BY
                    1.0
                    {self._reflection_order_factors()} DESC
                LIMIT ?1
            """
            rows = self.conn.execute(sql, (limit,)).fetchall()

        return self._rows_to_dicts(rows)

    def _merge_results(
        self, primary: list[dict], secondary: list[dict], limit: int
    ) -> list[dict]:
        """合并两路检索结果，去重后取 top limit。"""
        seen: set[int] = set()
        merged: list[dict] = []
        for r in primary:
            if r['id'] not in seen:
                seen.add(r['id'])
                merged.append(r)
        for r in secondary:
            if r['id'] not in seen and len(merged) < limit:
                seen.add(r['id'])
                merged.append(r)
        return merged[:limit]

    # ------------------------------------------------------------------
    # Tokenize / Query building
    # ------------------------------------------------------------------

    def _reflection_order_factors(self) -> str:
        """ORDER BY 片段：纯图拓扑信号回流到排序（无时间衰减，无锚定加成）。

        Tier 1 闭环 — 四个因子：
          - contradicted 惩罚：被指向的记忆 ×0.8（需 contradicted_enabled）
          - echoed 加成：有确认回声的记忆 ×1.1（需 echoed_enabled）
          - field 强度：场域记忆密度 → 对数缩放 ×1.0~1.15（始终生效）
          - belief 加成：作为高置信信念证据的记忆 ×1.0~1.15（Anchor 的结构替代）
        """
        parts = []
        if self.contradicted_enabled:
            parts.append(f"""        * COALESCE((SELECT {self.CONTRADICTED_PENALTY}
                    FROM reflections
                    WHERE target_id = m.id AND relation_type = 'contradicted'
                    LIMIT 1), 1.0)""")
        if self.echoed_enabled:
            parts.append(f"""        * COALESCE((SELECT {self.ECHOED_BONUS}
                    FROM reflections
                    WHERE (source_id = m.id OR target_id = m.id)
                      AND relation_type = 'echoed'
                    LIMIT 1), 1.0)""")
        parts.append(f"""        * COALESCE((SELECT MIN({self.FIELD_STRENGTH_CAP},
                               1.0 + LN(COUNT(*) + 1) * {self.FIELD_STRENGTH_SCALE})
                    FROM memories
                    WHERE field = m.field AND archived = 0), 1.0)""")
        # Belief 证据加成：结构替代 anchor——记忆的价值取决于它是否支撑当前 truth
        parts.append(f"""        * COALESCE((SELECT 1.0 + MAX(b.confidence - 0.5, 0.0) * 0.3
                    FROM beliefs b
                    WHERE b.field = m.field
                      AND b.superseded_by = 0
                      AND b.evidence_ids LIKE ('%' || m.id || '%')
                    LIMIT 1), 1.0)""")
        return "\n".join(parts) + "\n"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """简易中文词/短语提取：按标点切分后，取 bigram+ 子串作为词。"""
        cleaned = re.sub(r"[^\u4e00-\u9fff\w]", " ", text)
        words: set[str] = set()
        for seg in cleaned.split():
            seg = seg.strip()
            if len(seg) >= 2:
                for i in range(len(seg) - 1):
                    words.add(seg[i:i + 2])
                if len(seg) >= 3:
                    words.add(seg)
        return words

    def _escape_fts5(self, query: str) -> str:
        """转义 FTS5 特殊字符，保留有意义的词。"""
        result = []
        for ch in query:
            if ch.isalnum() or ch.isspace():
                result.append(ch)
            else:
                result.append(' ')
        cleaned = ''.join(result).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

    def _has_short_terms(self, query: str) -> bool:
        """检查查询是否包含 trigram 无法索引的短词（<3 字符）。"""
        cleaned = ''.join(
            ch if ch.isalnum() or ch.isspace() else ' '
            for ch in query
        ).strip()
        return any(len(t) < 3 for t in cleaned.split())

    def _build_fts5_query(self, query: str) -> str:
        """将用户查询转换为 FTS5 trigram 友好的 OR 查询。

        FTS5 trigram 的 AND 语义对中文不友好：查询中的所有 trigram
        都必须出现在目标文本中。通过将查询拆分为关键词并用 OR 连接，
        提高召回率。

        策略：
        - 清理标点，保留字母数字和空格
        - CJK 字符序列 → 3-4 字滑动窗口切分
        - 非 CJK token → 保留（2 字符以上）
        - 所有 term 加双引号用 OR 连接
        """
        cleaned = ''.join(
            ch if ch.isalnum() or ch.isspace() else ' '
            for ch in query
        ).strip()
        if not cleaned:
            return cleaned

        tokens = cleaned.split()
        terms = []

        for token in tokens:
            cjk_count = sum(1 for ch in token if '\u4e00' <= ch <= '\u9fff')
            if cjk_count > 0:
                if len(token) <= 4:
                    terms.append(token)
                else:
                    for i in range(len(token) - 2):
                        chunk = token[i:i+3]
                        if chunk not in terms:
                            terms.append(chunk)
                    if len(token) >= 6:
                        for i in range(len(token) - 3):
                            chunk = token[i:i+4]
                            if chunk not in terms:
                                terms.append(chunk)
            else:
                if len(token) >= 2:
                    terms.append(token)

        if not terms:
            return cleaned

        unique_terms = list(dict.fromkeys(terms))
        return ' OR '.join(f'"{t}"' for t in unique_terms)

    # ------------------------------------------------------------------
    # Field Snapshot — 动态场态计算
    # ------------------------------------------------------------------

    def field_snapshot(self, field: str) -> dict:
        """计算关系场的动态状态（"磁场"特征）。

        Returns:
            strength: 记忆密度 × 最近活跃度
            polarity: user/agent 痕迹比（>1 用户偏多，<1 Agent 偏多）
            center_of_gravity: 最近一条记忆的时间位置
            recent_reflections: 近 7 天新建关系边数
            top_relations: 强度最高的 5 条关系边
        """
        now = self._now()
        seven_days_ago = now - 7 * 86400

        total = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE field = ? AND archived = 0",
            (field,)
        ).fetchone()[0]

        user_count = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE field = ? AND source = 'user' AND archived = 0",
            (field,)
        ).fetchone()[0]

        agent_count = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE field = ? AND source = 'agent' AND archived = 0",
            (field,)
        ).fetchone()[0]

        last_memory = self.conn.execute(
            "SELECT id, content, source, created_at FROM memories "
            "WHERE field = ? AND archived = 0 ORDER BY created_at DESC LIMIT 1",
            (field,)
        ).fetchone()

        recent_reflections = self.conn.execute(
            "SELECT COUNT(*) FROM reflections r "
            "JOIN memories m1 ON r.source_id = m1.id "
            "JOIN memories m2 ON r.target_id = m2.id "
            "WHERE r.created_at >= ? AND (m1.field = ? OR m2.field = ?)",
            (seven_days_ago, field, field)
        ).fetchone()[0]

        top_relations = self.conn.execute(
            "SELECT r.id, r.source_id, r.target_id, r.relation_type, r.strength, "
            "m_s.content AS source_content, m_t.content AS target_content "
            "FROM reflections r "
            "JOIN memories m_s ON r.source_id = m_s.id "
            "JOIN memories m_t ON r.target_id = m_t.id "
            "WHERE m_s.field = ? OR m_t.field = ? "
            "ORDER BY r.strength DESC LIMIT 5",
            (field, field)
        ).fetchall()

        recency_factor = 0.0
        if last_memory:
            age_days = (now - last_memory["created_at"]) / 86400
            recency_factor = max(0.0, 1.0 - age_days / 30.0)

        strength = round(total * 0.3 + recency_factor * 0.7, 2)

        return {
            "field": field,
            "strength": strength,
            "total_memories": total,
            "user_count": user_count,
            "agent_count": agent_count,
            "polarity": round(user_count / max(1, agent_count), 2),
            "center_of_gravity": dict(last_memory) if last_memory else None,
            "recent_reflections": recent_reflections,
            "top_relations": self._rows_to_dicts(top_relations),
            "computed_at": now,
        }

    # ------------------------------------------------------------------
    # Management API
    # ------------------------------------------------------------------

    def check_silence(self, days: int = 90) -> list[dict]:
        """查找静默记忆。

        静默条件：
          - created_at 超过 `days` 天
          - accessed_at 超过 `days` 天
          - anchor = 0

        Returns:
            静默记忆列表，按 field 分组。
        """
        threshold = self._now() - (days * 86400)
        sql = """
            SELECT * FROM memories
            WHERE created_at < ?
              AND accessed_at < ?
              AND anchor = 0
              AND archived = 0
            ORDER BY field, accessed_at DESC
        """
        rows = self.conn.execute(sql, (threshold, threshold)).fetchall()
        return self._rows_to_dicts(rows)

    def check_silence_by_field(self, days: int = 90) -> dict[str, list[dict]]:
        """按场分组返回静默记忆。

        Returns:
            {field_name: [memory_dict, ...], ...}
        """
        silent = self.check_silence(days)
        by_field: dict[str, list[dict]] = {}
        for mem in silent:
            f = mem.get("field") or "__no_field__"
            by_field.setdefault(f, []).append(mem)
        return by_field

    def get_memory(self, memory_id: int) -> Optional[dict]:
        """获取单条记忆。"""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_fields(self) -> list[dict]:
        """列出所有关系场及其记忆数量和极性。"""
        rows = self.conn.execute("""
            SELECT
                field,
                COUNT(*) AS count,
                SUM(anchor) AS anchored_count,
                SUM(archived) AS archived_count,
                SUM(CASE WHEN source = 'user' THEN 1 ELSE 0 END) AS user_count,
                SUM(CASE WHEN source = 'agent' THEN 1 ELSE 0 END) AS agent_count,
                MAX(accessed_at) AS last_touched
            FROM memories
            GROUP BY field
            ORDER BY last_touched DESC
        """).fetchall()
        return self._rows_to_dicts(rows)

    def list_by_field(self, field: str, include_archived: bool = False) -> list[dict]:
        """列出指定场下的所有记忆。"""
        if include_archived:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE field = ? ORDER BY accessed_at DESC",
                (field,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE field = ? AND archived = 0 ORDER BY accessed_at DESC",
                (field,)
            ).fetchall()
        return self._rows_to_dicts(rows)

    def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """纯 FTS5 全文搜索。"""
        fts_query = self._escape_fts5(query)
        if not fts_query:
            return []
        sql = """
            SELECT m.*, f.rank
            FROM memories m
            JOIN memory_fts f ON m.id = f.rowid
            WHERE memory_fts MATCH ?
            ORDER BY f.rank
            LIMIT ?
        """
        rows = self.conn.execute(sql, (fts_query, limit)).fetchall()
        return self._rows_to_dicts(rows)

    def stats(self) -> dict:
        """数据库统计信息。"""
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        anchored = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE anchor = 1"
        ).fetchone()[0]
        archived = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 1"
        ).fetchone()[0]
        fields = self.conn.execute(
            "SELECT COUNT(DISTINCT field) FROM memories WHERE field IS NOT NULL"
        ).fetchone()[0]
        total_reflections = self.conn.execute(
            "SELECT COUNT(*) FROM reflections"
        ).fetchone()[0]
        user_count = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE source = 'user'"
        ).fetchone()[0]
        agent_count = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE source = 'agent'"
        ).fetchone()[0]
        return {
            "total": total,
            "anchored": anchored,
            "archived": archived,
            "active": total - archived,
            "fields": fields,
            "user_memories": user_count,
            "agent_memories": agent_count,
            "total_reflections": total_reflections,
            "db_path": self.db_path,
        }

    def debug_vocab(
        self, top_n: int = 30, query: Optional[str] = None
    ) -> dict:
        """调试工具：查看 FTS5 索引词表。

        创建临时 FTS5 vocab 表，分析：
          - 索引中最常见的 token
          - 短 token（<3 字符，trigram 会丢弃）
          - 可选：某条查询的 term 在索引中的覆盖情况

        Args:
            top_n: 展示前 N 个高频 token。
            query: 可选，分析此查询的 term 覆盖率。

        Returns:
            包含 total_terms, top_terms, short_terms, query_analysis 的 dict。
        """
        result = {"total_terms": 0, "top_terms": [], "short_terms": []}

        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _dyadcore_debug_vocab "
                "USING fts5vocab('memory_fts', 'instance')"
            )

            agg_sql = """
                SELECT term, COUNT(*) AS cnt, COUNT(DISTINCT doc) AS docs
                FROM _dyadcore_debug_vocab
                GROUP BY term
            """
            agg_rows = self.conn.execute(
                f"{agg_sql} ORDER BY cnt DESC LIMIT ?", (top_n,)
            ).fetchall()
            result["total_terms"] = self.conn.execute(
                "SELECT COUNT(DISTINCT term) FROM _dyadcore_debug_vocab"
            ).fetchone()[0]

            for r in agg_rows:
                result["top_terms"].append({
                    "term": r[0], "count": r[1], "docs": r[2]
                })

            short_rows = self.conn.execute(
                f"{agg_sql} HAVING length(term) < 3 ORDER BY cnt DESC LIMIT ?",
                (top_n,)
            ).fetchall()
            for r in short_rows:
                result["short_terms"].append({
                    "term": r[0], "count": r[1], "docs": r[2]
                })

            if query:
                fts_query = self._build_fts5_query(query)
                query_terms = fts_query.split(' OR ') if fts_query else []
                query_terms = [t.strip('"') for t in query_terms]
                term_hits = {}
                for t in query_terms:
                    cnt = self.conn.execute(
                        "SELECT COUNT(*) FROM _dyadcore_debug_vocab "
                        "WHERE term = ?", (t,)
                    ).fetchone()[0]
                    term_hits[t] = cnt
                result["query_analysis"] = {
                    "query": query,
                    "fts5_query": fts_query,
                    "terms": list(term_hits.items()),
                    "hit_count": sum(1 for v in term_hits.values() if v > 0),
                    "total_terms": len(term_hits),
                    "coverage": (
                        sum(1 for v in term_hits.values() if v > 0) / len(term_hits)
                        if term_hits else 0
                    ),
                }

        finally:
            self.conn.execute("DROP TABLE IF EXISTS _dyadcore_debug_vocab")

        return result

    def close(self) -> None:
        """关闭数据库连接。"""
        self.conn.close()
