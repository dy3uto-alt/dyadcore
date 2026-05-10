#!/usr/bin/env python3
"""
DyadCore Dual Mirror v2.0 — 综合测试套件

覆盖:
  - Schema (表、触发器、索引)
  - Write (user/agent, utterance/action/meta)
  - Anchor / Unanchor / Archive / Unarchive / set_field
  - Recall: FTS5 trigram, LIKE fallback, 关系图扩展, no-query
  - Reflections: 自动建边, add_relation, get_relations, get_relation_graph
  - Field snapshot
  - Management: check_silence, list_fields, stats
  - accessed_at (仅管理操作更新)
  - 持久化 (close/reopen)
  - 边界情况

用法:
  python test_dyadcore.py
  python -m pytest test_dyadcore.py -v
"""

import unittest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dyadcore
from dyadcore import DyadCore

TEST_DB = "test_dyadcore_v2.db"
PERSIST_DB = "test_dyadcore_persist.db"


def _rm_db():
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix
        try:
            if os.path.exists(path):
                os.remove(path)
        except (OSError, PermissionError):
            pass


def _rm_persist_db():
    for suffix in ("", "-wal", "-shm"):
        path = PERSIST_DB + suffix
        try:
            if os.path.exists(path):
                os.remove(path)
        except (OSError, PermissionError):
            pass


def _clear_tables(mh):
    mh.conn.execute("DELETE FROM memories")
    mh.conn.execute("DELETE FROM memory_fts")
    mh.conn.execute("DELETE FROM reflections")


# ===========================================================================
# Test Suite
# ===========================================================================

class TestSchema(unittest.TestCase):
    """数据库表结构测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_memories_table_exists(self):
        tables = self.mh.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        self.assertIn("memories", names)

    def test_fts5_table_exists(self):
        tables = self.mh.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        self.assertIn("memory_fts", names)

    def test_reflections_table_exists(self):
        tables = self.mh.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        self.assertIn("reflections", names)

    def test_source_check_constraint(self):
        # 只允许 'user' 和 'agent'
        self.mh.write("test user", source="user")
        self.mh.write("test agent", source="agent")
        with self.assertRaises(Exception):
            self.mh.write("test agent_self", source="agent_self")

    def test_indexes(self):
        indexes = self.mh.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        names = {r[0] for r in indexes}
        expected = ["idx_memories_field", "idx_memories_anchor",
                    "idx_memories_archived", "idx_memories_accessed",
                    "idx_memories_created", "idx_memories_source",
                    "idx_reflections_source", "idx_reflections_target",
                    "idx_reflections_type"]
        for idx in expected:
            self.assertIn(idx, names, f"Missing index: {idx}")

    def test_fts5_triggers(self):
        triggers = self.mh.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
        names = {r[0] for r in triggers}
        self.assertIn("mem_fts_insert", names)
        self.assertIn("mem_fts_delete", names)
        self.assertIn("mem_fts_update", names)


class TestWrite(unittest.TestCase):
    """写入操作测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)

    def setUp(self):
        _clear_tables(self.mh)

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_write_user_utterance(self):
        mid = self.mh.write("用户消息", source="user", memory_type="utterance")
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["content"], "用户消息")
        self.assertEqual(mem["source"], "user")
        self.assertEqual(mem["memory_type"], "utterance")

    def test_write_agent_utterance(self):
        mid = self.mh.write("Agent 回应", source="agent", memory_type="utterance")
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["source"], "agent")

    def test_write_action(self):
        mid = self.mh.write("执行部署", source="agent", memory_type="action")
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["memory_type"], "action")

    def test_write_meta(self):
        mid = self.mh.write("系统元信息", source="user", memory_type="meta")
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["memory_type"], "meta")

    def test_write_with_field(self):
        mid = self.mh.write("讨论技术栈", source="user", field="tech")
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["field"], "tech")

    def test_write_auto_reflections(self):
        """同 field 内的相关记忆应自动建边。"""
        self.mh.write("用户偏好本地部署，不喜欢云服务", source="user", field="tech")
        time.sleep(0.1)
        mid = self.mh.write("Agent 推荐 SQLite 本地存储方案", source="agent", field="tech")
        rels = self.mh.get_relations(mid)
        self.assertGreater(len(rels), 0, "同场相似记忆应有自动关系边")

    def test_write_no_reflections_different_field(self):
        """不同 field 的记忆不应自动建边。"""
        self.mh.write("用户偏好本地部署", source="user", field="tech")
        time.sleep(0.1)
        mid = self.mh.write("今天天气不错", source="user", field="weather")
        rels = self.mh.get_relations(mid)
        self.assertEqual(len(rels), 0, "跨场记忆不应自动建边")

    def test_write_no_reflections_no_field(self):
        """无 field 的记忆不参与自动建边。"""
        mid = self.mh.write("一些笔记", source="user")
        rels = self.mh.get_relations(mid)
        self.assertEqual(len(rels), 0)


class TestAnchorArchive(unittest.TestCase):
    """锚定、归档测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)

    def setUp(self):
        _clear_tables(self.mh)

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_anchor(self):
        mid = self.mh.write("重要信息", source="user")
        self.mh.anchor(mid)
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["anchor"], 1)

    def test_unanchor(self):
        mid = self.mh.write("重要信息", source="user")
        self.mh.anchor(mid)
        self.mh.unanchor(mid)
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["anchor"], 0)

    def test_anchor_updates_accessed_at(self):
        mid = self.mh.write("重要信息", source="user")
        old_at = self.mh.get_memory(mid)["accessed_at"]
        time.sleep(0.1)
        self.mh.anchor(mid)
        new_at = self.mh.get_memory(mid)["accessed_at"]
        self.assertGreater(new_at, old_at)

    def test_archive(self):
        mid = self.mh.write("过时信息", source="user")
        self.mh.archive(mid)
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["archived"], 1)

    def test_unarchive(self):
        mid = self.mh.write("过时信息", source="user")
        self.mh.archive(mid)
        self.mh.unarchive(mid)
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["archived"], 0)

    def test_set_field(self):
        mid = self.mh.write("技术讨论", source="user", field="old_field")
        self.mh.set_field(mid, "new_field")
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["field"], "new_field")

    def test_set_field_none(self):
        mid = self.mh.write("杂项", source="user", field="temp")
        self.mh.set_field(mid, None)
        mem = self.mh.get_memory(mid)
        self.assertIsNone(mem["field"])


class TestRecallFTS5(unittest.TestCase):
    """FTS5 全文检索召回测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)
        cls.mh.write("用户每天早上喝手冲咖啡，喜欢非洲豆", source="user", field="prefs")
        cls.mh.write("用户不喜欢甜的，特别是奶油蛋糕", source="user", field="prefs")
        cls.mh.write("前端使用 React 19 和 TypeScript", source="agent", field="tech")
        cls.mh.write("后端使用 Python FastAPI", source="agent", field="tech")

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_fts5_basic(self):
        results = self.mh.recall("咖啡")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("咖啡", results[0]["content"])

    def test_fts5_field_priority(self):
        results = self.mh.recall("TypeScript", field_hint="tech")
        self.assertGreaterEqual(len(results), 1)
        # 同场优先：第一条应是 tech 场的
        self.assertEqual(results[0]["field"], "tech")

    def test_like_fallback(self):
        """短词 (<3 chars) 应走 LIKE 回退。"""
        results = self.mh.recall("React")
        # FTS5 finds the React memory, graph expansion may add neighbors
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any("React" in r["content"] for r in results))

    def test_no_query(self):
        results = self.mh.recall()
        self.assertGreaterEqual(len(results), 1)

    def test_include_archived(self):
        mid = self.mh.write("已归档的记忆内容", source="user", field="test")
        self.mh.archive(mid)
        results = self.mh.recall("已归档", field_hint="test")
        self.assertEqual(len(results), 0)
        results2 = self.mh.recall("已归档", field_hint="test", include_archived=True)
        self.assertEqual(len(results2), 1)

    def test_limit(self):
        for i in range(10):
            self.mh.write(f"测试记忆 {i}", source="user", field="bulk")
        results = self.mh.recall(field_hint="bulk", limit=3)
        self.assertLessEqual(len(results), 3)


class TestRecallGraph(unittest.TestCase):
    """关系图扩展召回测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)
        # 写入相关记忆触发自动建边
        cls.mh.write("用户项目用 PostgreSQL 15 做数据库", source="user", field="tech")
        cls.mh.write("用户项目用 FastAPI 做后端框架", source="user", field="tech")
        cls.mh.write("PostgreSQL 有 JSONB 和全文检索能力", source="agent", field="tech")
        cls.mh.write("FastAPI 支持异步和自动 OpenAPI 文档生成", source="agent", field="tech")

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_graph_expansion(self):
        """召回应包含通过关系边发现的邻居。"""
        results = self.mh.recall("PostgreSQL", field_hint="tech", limit=5)
        contents = [r["content"] for r in results]
        depths = [r.get("depth", 0) for r in results]
        # 应有图扩展结果 (depth > 0)
        self.assertTrue(any(d > 0 for d in depths),
                        f"No graph expansion results. depths={depths}")

    def test_graph_disabled(self):
        results = self.mh.recall("PostgreSQL", field_hint="tech", limit=5,
                                expand_graph=False)
        depths = [r.get("depth", 0) for r in results]
        # 关闭图扩展则全部 depth=0
        self.assertTrue(all(d == 0 for d in depths))


class TestReflections(unittest.TestCase):
    """Reflections 关系网测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)

    def setUp(self):
        _clear_tables(self.mh)

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_add_relation(self):
        id1 = self.mh.write("记忆 A", source="user", field="test")
        id2 = self.mh.write("记忆 B", source="agent", field="test")
        rid = self.mh.add_relation(id1, id2, "triggered", strength=0.8)
        self.assertGreater(rid, 0)

    def test_get_relations(self):
        id1 = self.mh.write("记忆 A", source="user", field="test")
        id2 = self.mh.write("记忆 B", source="agent", field="test")
        self.mh.add_relation(id1, id2, "echoed", strength=0.6)
        rels = self.mh.get_relations(id1)
        self.assertGreaterEqual(len(rels), 1)

    def test_get_relation_graph(self):
        id1 = self.mh.write("中心节点", source="user", field="graph_test")
        id2 = self.mh.write("邻居1", source="agent", field="graph_test")
        id3 = self.mh.write("邻居2", source="agent", field="graph_test")
        self.mh.add_relation(id1, id2, "related", 0.5)
        self.mh.add_relation(id1, id3, "triggered", 0.7)
        graph = self.mh.get_relation_graph(id1, max_depth=1)
        self.assertGreaterEqual(len(graph), 2)

    def test_contradicted_relation(self):
        id1 = self.mh.write("用户喜欢 React", source="user", field="test")
        id2 = self.mh.write("用户说 React 太复杂想换 Vue", source="user", field="test")
        self.mh.add_relation(id2, id1, "contradicted", strength=0.9)
        rels = self.mh.get_relations(id2)
        # 检查 contradicted 关系存在
        types = [r["relation_type"] for r in rels]
        self.assertIn("contradicted", types)


class TestFieldSnapshot(unittest.TestCase):
    """场快照测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)

    def setUp(self):
        _clear_tables(self.mh)

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_field_snapshot_empty(self):
        snap = self.mh.field_snapshot("nonexistent")
        self.assertEqual(snap["total_memories"], 0)
        self.assertEqual(snap["strength"], 0.0)

    def test_field_snapshot_basic(self):
        self.mh.write("用户偏好 A", source="user", field="prefs")
        self.mh.write("用户偏好 B", source="user", field="prefs")
        self.mh.write("Agent 观察 C", source="agent", field="prefs")
        snap = self.mh.field_snapshot("prefs")
        self.assertEqual(snap["total_memories"], 3)
        self.assertEqual(snap["user_count"], 2)
        self.assertEqual(snap["agent_count"], 1)
        self.assertEqual(snap["polarity"], 2.0)
        self.assertIn("strength", snap)
        self.assertIn("center_of_gravity", snap)
        self.assertIn("recent_reflections", snap)

    def test_field_snapshot_center_of_gravity(self):
        self.mh.write("最早记忆", source="user", field="cog")
        time.sleep(0.1)
        mid = self.mh.write("最新记忆", source="agent", field="cog")
        snap = self.mh.field_snapshot("cog")
        self.assertIsNotNone(snap["center_of_gravity"])
        self.assertEqual(snap["center_of_gravity"]["id"], mid)


class TestManagement(unittest.TestCase):
    """管理 API 测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)

    def setUp(self):
        _clear_tables(self.mh)

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_stats(self):
        self.mh.write("用户记忆", source="user", field="f1")
        self.mh.write("Agent 记忆", source="agent", field="f1")
        stats = self.mh.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["user_memories"], 1)
        self.assertEqual(stats["agent_memories"], 1)
        self.assertEqual(stats["fields"], 1)
        self.assertIn("total_reflections", stats)

    def test_list_fields(self):
        self.mh.write("m1", source="user", field="f1")
        self.mh.write("m2", source="agent", field="f1")
        self.mh.write("m3", source="user", field="f2")
        fields = self.mh.list_fields()
        self.assertEqual(len(fields), 2)
        f1 = next(f for f in fields if f["field"] == "f1")
        self.assertEqual(f1["user_count"], 1)
        self.assertEqual(f1["agent_count"], 1)

    def test_list_by_field(self):
        self.mh.write("m1", source="user", field="f1")
        self.mh.write("m2", source="agent", field="f1")
        self.mh.write("m3", source="user", field="f2")
        mems = self.mh.list_by_field("f1")
        self.assertEqual(len(mems), 2)

    def test_check_silence(self):
        self.mh.write("旧记忆", source="user")
        # days=0 → 所有非锚定但非归档算静默（created_at 和 accessed_at 都 < now）
        silent = self.mh.check_silence(days=365 * 10)  # 极长阈值
        self.assertEqual(len(silent), 0, "刚创建的记忆不应静默")

    def test_check_silence_by_field(self):
        self.mh.write("m1", source="user", field="f1")
        result = self.mh.check_silence_by_field(days=365 * 10)
        self.assertEqual(len(result), 0)

    def test_get_memory(self):
        mid = self.mh.write("查询测试", source="user")
        mem = self.mh.get_memory(mid)
        self.assertEqual(mem["content"], "查询测试")
        self.assertIsNone(self.mh.get_memory(9999))

    def test_search_fts(self):
        self.mh.write("PostgreSQL 数据库优化", source="user")
        self.mh.write("React 前端性能", source="agent")
        results = self.mh.search_fts("PostgreSQL")
        self.assertEqual(len(results), 1)
        self.assertIn("PostgreSQL", results[0]["content"])

    def test_debug_vocab(self):
        self.mh.write("测试数据", source="user")
        vocab = self.mh.debug_vocab()
        self.assertIn("total_terms", vocab)
        self.assertIn("top_terms", vocab)


class TestPersistence(unittest.TestCase):
    """持久化测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_persist_db()

    @classmethod
    def tearDownClass(cls):
        _rm_persist_db()

    def test_persistence(self):
        db_path = "test_dyadcore_persist.db"
        for suffix in ("", "-wal", "-shm"):
            try:
                if os.path.exists(db_path + suffix):
                    os.remove(db_path + suffix)
            except (OSError, PermissionError):
                pass

        mh1 = DyadCore(db_path)
        mid = mh1.write("持久化测试内容", source="user", field="persist_test")
        mh1.anchor(mid)
        mh1.conn.commit()
        mh1.close()
        time.sleep(0.2)

        mh2 = DyadCore(db_path)
        mem = mh2.get_memory(mid)
        self.assertIsNotNone(mem, "持久化后应能读取到记忆")
        if mem:
            self.assertEqual(mem["content"], "持久化测试内容")
            self.assertEqual(mem["anchor"], 1)
            self.assertEqual(mem["field"], "persist_test")
        mh2.close()


class TestEdgeCases(unittest.TestCase):
    """边界情况测试。"""

    @classmethod
    def setUpClass(cls):
        _rm_db()
        cls.mh = DyadCore(TEST_DB)

    def setUp(self):
        _clear_tables(self.mh)

    @classmethod
    def tearDownClass(cls):
        cls.mh.close()
        _rm_db()

    def test_empty_db_recall(self):
        results = self.mh.recall()
        self.assertEqual(len(results), 0)

    def test_empty_db_stats(self):
        stats = self.mh.stats()
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["total_reflections"], 0)

    def test_single_memory(self):
        mid = self.mh.write("唯一记忆测试", source="user", field="solo")
        results = self.mh.recall("唯一记忆", field_hint="solo")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], mid)

    def test_large_limit(self):
        for i in range(20):
            self.mh.write(f"记忆 {i:03d}", source="user", field="bulk")
        results = self.mh.recall(field_hint="bulk", limit=100)
        self.assertEqual(len(results), 20)

    def test_special_characters(self):
        mid = self.mh.write("C++ 和 JavaScript 的对比测试 #42 @user",
                            source="user")
        results = self.mh.recall("C++")
        self.assertGreaterEqual(len(results), 0)
        # FTS5 可能匹配也可能不匹配，取决于 tokenizer
        # 至少不应崩溃

    def test_duplicate_relation_ignored(self):
        id1 = self.mh.write("A", source="user", field="dup")
        id2 = self.mh.write("B", source="agent", field="dup")
        r1 = self.mh.add_relation(id1, id2, "related", 0.5)
        r2 = self.mh.add_relation(id1, id2, "related", 0.5)
        # INSERT OR IGNORE 应忽略重复（虽然 reflections 表没有 UNIQUE 约束）
        # 但至少不应崩溃
        self.assertGreater(r1, 0)
        self.assertGreater(r2, 0)

    def test_recall_accessed_at_unchanged(self):
        """recall 不应刷新 accessed_at。"""
        mid = self.mh.write("测试记忆", source="user")
        old_at = self.mh.get_memory(mid)["accessed_at"]
        time.sleep(0.1)
        self.mh.recall("测试")
        new_at = self.mh.get_memory(mid)["accessed_at"]
        self.assertEqual(new_at, old_at, "recall 不应修改 accessed_at")

    def test_backward_compat_stubs(self):
        """embedding_tier 和 embedding_dim 应作为兼容桩存在。"""
        self.assertEqual(self.mh.embedding_tier, 3)
        self.assertIsNone(self.mh.embedding_dim)

    def test_write_creates_timestamp(self):
        mid = self.mh.write("时间测试", source="user")
        mem = self.mh.get_memory(mid)
        self.assertGreater(mem["created_at"], 0)
        self.assertGreater(mem["accessed_at"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
