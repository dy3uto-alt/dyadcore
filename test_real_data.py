#!/usr/bin/env python3
"""
DyadCore Dual Mirror 测试 — 模拟 5 轮对话，
验证 FTS5 召回、场域隔离、reflections 关系图、field_snapshot。
"""

import os
import time
from dyadcore import DyadCore

# 清理旧数据库
if os.path.exists("test_real.db"):
    os.remove("test_real.db")

mh = DyadCore("test_real.db")

print("=" * 60)
print("DyadCore Dual Mirror 测试")
print("=" * 60)

# ========== 第 1 轮：初始化场域 ==========
print("\n【第 1 轮】用户启动本地记忆选型讨论")

uid1 = mh.write("用户说：我想做一个本地记忆系统，不用 Docker，不用 Postgres",
                memory_type="utterance", source="user", field="本地记忆选型")
uid2 = mh.write("用户说：sqlite-vec 看起来不错，单文件",
                memory_type="utterance", source="user", field="本地记忆选型")
aid1 = mh.write("Agent 回应：sqlite-vec 是 Mozilla 的 SQLite 扩展，零依赖",
                memory_type="utterance", source="agent", field="本地记忆选型")

print(f"  写入 3 条记忆，id={uid1},{uid2},{aid1}")

# ========== 第 2 轮：Agent 写入观察（对等的 agent 痕迹）==========
print("\n【第 2 轮】Agent 基于对话写入观察")

sid1 = mh.write("用户厌恶多服务架构，偏好单文件方案，这是长期认知",
                memory_type="utterance", source="agent", field="本地记忆选型")
mem = mh.get_memory(sid1)
print(f"  写入 agent 观察 id={sid1}, source='{mem['source']}', field='{mem['field']}'")

# 查看关系
rels = mh.get_relations(sid1)
print(f"  自动关系边: {len(rels)} 条")
for r in rels:
    print(f"    [{r['relation_type']}] strength={r['strength']:.3f} "
          f"<-> '{r.get('target_content', '')[:40]}...'")

# ========== 第 3 轮：场域切换，测试跨场隔离 ==========
print("\n【第 3 轮】用户切换到前端话题")

uid3 = mh.write("用户说：前端用 React 19 还是 Vue 3？",
                memory_type="utterance", source="user", field="前端重构")
aid2 = mh.write("Agent 回应：React 19 有 Server Components，适合你的本地优先哲学",
                memory_type="utterance", source="agent", field="前端重构")

print(f"  写入 2 条前端记忆，id={uid3},{aid2}")

# 测试：从"前端重构"场召回，同场优先
print("\n  >>> 召回测试：field_hint='前端重构'，limit=5")
results = mh.recall(query="React", field_hint="前端重构", limit=5)
for i, r in enumerate(results, 1):
    bucket = "同场" if r.get('bucket') == 0 else "跨场"
    depth_tag = f" [graph depth={r.get('depth',0)}]" if r.get('depth', 0) > 0 else ""
    print(f"  {i}. [{bucket}{depth_tag}] | {r['source']:6s} | {r['content'][:50]}...")

# ========== 第 4 轮：测试 accessed_at 不刷新 ==========
print("\n【第 4 轮】验证 accessed_at 行为")

old_accessed = mh.get_memory(uid1)['accessed_at']
time.sleep(1.1)  # 确保时间差

# 召回 uid1 所在的场，但不操作它
mh.recall(query="sqlite-vec", field_hint="本地记忆选型", limit=3)
new_accessed = mh.get_memory(uid1)['accessed_at']

print(f"  uid1 旧 accessed_at: {old_accessed:.0f}")
print(f"  uid1 新 accessed_at: {new_accessed:.0f}")
print(f"  是否变化: {'[BUG!]' if new_accessed != old_accessed else '[OK] 正确（recall 不刷新）'}")

# ========== 第 5 轮：归档 + 静默期测试 ==========
print("\n【第 5 轮】归档旧记忆，测试静默期")

mh.archive(uid2)  # 归档"sqlite-vec 看起来不错"
print(f"  归档 uid2={uid2}")

# 召回本地记忆选型场，确认归档的不出现
print("\n  >>> 召回测试：field_hint='本地记忆选型'，limit=5")
results = mh.recall(field_hint="本地记忆选型", limit=5)
print(f"  召回 {len(results)} 条（归档记忆应被过滤）")
for i, r in enumerate(results, 1):
    status = "归档" if r['archived'] else "活跃"
    depth_tag = f" [graph depth={r.get('depth',0)}]" if r.get('depth', 0) > 0 else ""
    print(f"  {i}. [{status}{depth_tag}] | {r['content'][:50]}...")

# 静默期检测
print("\n  >>> 静默期检测（days=0，即所有非锚定都算静默）")
silent = mh.check_silence(days=0)
print(f"  静默记忆数: {len(silent)}（应包含 uid1/uid3/aid2）")

# ========== 场快照 ==========
print("\n--- 场快照 ---")
for f_name in ["本地记忆选型", "前端重构"]:
    snap = mh.field_snapshot(f_name)
    if snap['total_memories'] > 0:
        print(f"  {f_name}: 强度={snap['strength']}, "
              f"极性={snap['polarity']} (用户{snap['user_count']}/Agent{snap['agent_count']}), "
              f"关系={snap['recent_reflections']}")

# ========== 关系图测试 ==========
print("\n--- 关系图测试 ---")
graph = mh.get_relation_graph(sid1, max_depth=1)
print(f"  id={sid1} 的 1-hop 关系子图 ({len(graph)} 条边):")
for edge in graph[:5]:
    print(f"    [{edge['relation_type']}] depth={edge['depth']} "
          f"strength={edge['strength']:.3f}")

# ========== 统计 ==========
print("\n" + "=" * 60)
print("最终统计")
print("=" * 60)
stats = mh.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n场域分布:")
for f in mh.list_fields():
    print(f"  {f['field'] or '(无场)'}: {f['count']}条 (锚定{f['anchored_count']}, 用户{f['user_count']}, Agent{f['agent_count']})")

mh.close()
print("\n[OK] Dual Mirror 测试完成")
