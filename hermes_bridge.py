#!/usr/bin/env python3
"""
HermesBridge — DyadCore 与对话管线的胶水层

提供三个辅助函数和一个模拟对话脚本：
  format_for_prompt()     将 recall 结果格式化为 prompt 上下文
  should_recall()         三触发条件检测（回溯词/名词漂移/冷场）
  should_write_agent_self()  判断本轮是否写入 agent_self 反思
"""

import json
import re
from typing import Optional

# ============================================================
# 回溯词表
# ============================================================
RETROSPECTIVE_KEYWORDS = [
    "上次", "之前", "上次我们", "之前我们", "还记得", "之前说到",
    "继续", "接着", "刚才", "前面", "刚刚", "你之前说",
    "回顾", "回到", "再说说", "再讲讲", "上次那个",
    "之前聊", "上次讨论", "之前讨论", "接着说", "然后呢",
    "后来呢", "再之前", "更早", "那时", "那天",
    "你提到过", "你说过", "我们聊过",
]


# ============================================================
# 中文分词辅助（简易版，无外部依赖）
# ============================================================
def _tokenize(text: str) -> set[str]:
    """简易中文词/短语提取：按标点切分后，取长度>=2的子串作为词。"""
    # 去掉标点和空格
    cleaned = re.sub(r"[^\u4e00-\u9fff\w]", " ", text)
    words = set()
    for seg in cleaned.split():
        seg = seg.strip()
        if len(seg) >= 2:
            # 单字无意义，按 bigram+ 切
            for i in range(len(seg) - 1):
                words.add(seg[i:i + 2])
            if len(seg) >= 3:
                words.add(seg)  # 全段也作为一个词
    return words


# ============================================================
# 1. format_for_prompt
# ============================================================
def format_for_prompt(results: list, dyadcore=None, title: str = "关系场背景") -> str:
    """将 recall 结果格式化为可注入 system prompt 的字符串。

    当提供 dyadcore 实例时，会自动检测结果集中的 contradicted 关系边，
    将新旧版本的演化链以拓扑形式渲染（旧版缩进标注"已更新"）。

    Args:
        results: DyadCore.recall() 返回的 dict 列表
        dyadcore: 可选 DyadCore 实例，用于查询 contradicted 关系边
        title: 标题行文字

    Returns:
        格式化后的字符串，无结果时返回空字符串
    """
    if not results:
        return ""

    # --- 检测 contradicted 关系边（通过公开 API，不再直接访问 .conn）---
    contradicted_map: dict[int, int] = {}  # old_id -> new_id
    if dyadcore is not None and len(results) >= 2:
        result_ids = [r['id'] for r in results]
        contradicted_map = dyadcore.get_contradicted_map(result_ids)

    lines = [f"## {title}（仅供参考，勿直接复述）", ""]

    # --- 分组：演化链 vs 独立记忆 ---
    chained_old = set(contradicted_map.keys())
    chained_new = set(contradicted_map.values())
    standalone = [r for r in results
                  if r['id'] not in chained_old and r['id'] not in chained_new]
    chains = list(contradicted_map.items())  # [(old_id, new_id), ...]

    # --- 渲染独立记忆 ---
    num = 1
    for r in standalone:
        lines.append(_format_single(r, num))
        num += 1

    # --- 渲染演化链 ---
    if chains:
        if standalone:
            lines.append("")
        lines.append("[演化链 — 以下为当前版本，旧版仅作废弃参考]")
        for old_id, new_id in chains:
            old_mem = next((r for r in results if r['id'] == old_id), None)
            new_mem = next((r for r in results if r['id'] == new_id), None)
            if old_mem and new_mem:
                lines.append(_format_single(new_mem, num))
                # 旧版截断为一行摘要，明确标注已过时
                old_content = old_mem.get('content', '')[:80]
                lines.append(f"   ~~ 旧版(已过时): {old_content}")
                num += 1

    return "\n".join(lines)


def _format_single(r: dict, num: int) -> str:
    """格式化单条记忆为一行。"""
    anchor_tag = " [锚定]" if r.get("anchor") else ""
    field = r.get("field") or "(无场)"
    content = r.get("content", "")
    source = r.get("source", "unknown")
    via = ""
    if r.get('depth', 0) > 0:
        via = f" [graph/{r.get('via_type', '')}]"
    return f"{num}. [{source} | {field}{anchor_tag}{via}] {content}"


# ============================================================
# 2. should_recall
# ============================================================
def should_recall(
    history: list,
    *,
    recent_turns: int = 3,
    drift_threshold: float = 0.4,
) -> bool:
    """三触发条件检测：回溯词 / 名词漂移 / 冷场。

    Args:
        history: 对话历史列表，每项为 {"role": "user"|"agent", "content": str}
        recent_turns: 检测名词漂移时，对比的最近轮数
        drift_threshold: 漂移比例阈值（新词占比超过此值即触发）

    Returns:
        是否应该执行 recall
    """
    # --- 冷场：空历史 ---
    if not history:
        return True

    # 只取用户消息
    user_msgs = [h["content"] for h in history if h.get("role") == "user"]
    if not user_msgs:
        return True

    last_msg = user_msgs[-1]

    # --- 回溯词检测 ---
    for kw in RETROSPECTIVE_KEYWORDS:
        if kw in last_msg:
            return True

    # --- 名词漂移检测 ---
    # 对比最近一条 vs 前 recent_turns 条的词集合
    recent_window = user_msgs[-recent_turns:-1] if len(user_msgs) >= 2 else []
    if recent_window:
        old_words = set()
        for msg in recent_window:
            old_words |= _tokenize(msg)

        new_words = _tokenize(last_msg)

        if new_words:
            overlap = new_words & old_words
            drift_ratio = 1.0 - len(overlap) / len(new_words)
            if drift_ratio >= drift_threshold:
                return True

    return False


# ============================================================
# 3. should_write_agent_self
# ============================================================
# agent_self 触发特征词
SELF_REFLECTION_CUES = [
    "偏好", "喜欢", "不喜欢", "讨厌", "希望", "想要", "要求",
    "习惯", "总是", "从不", "必须", "一定", "重要",
    "原则", "底线", "长期", "一直", "坚持",
    "我决定", "我选择", "我倾向", "我计划", "我打算",
    "我觉得", "我认为", "我的", "我需要",
    "以后", "今后", "下次", "未来",
]

# 每轮最多写 1 条，整个会话最多 10 条
MAX_AGENT_SELF_PER_SESSION = 10


def should_write_agent_self(
    history: list,
    agent_self_count: int,
    *,
    max_count: int = MAX_AGENT_SELF_PER_SESSION,
    force_every_n_turns: int = 6,
) -> bool:
    """判断当前轮是否应该写入 agent_self 反思。

    触发条件（满足任一）：
    1. 用户最新消息包含自我反思线索词
    2. 距离上次 agent_self 已过 force_every_n_turns 轮

    Args:
        history: 对话历史列表
        agent_self_count: 当前会话已写入的 agent_self 数量
        max_count: 上限（默认 10）
        force_every_n_turns: 强制写入间隔（默认 6 轮）

    Returns:
        是否应该写入 agent_self
    """
    if agent_self_count >= max_count:
        return False

    if not history:
        return False

    # 取最后一轮用户消息
    user_msgs = [h for h in history if h.get("role") == "user"]
    if not user_msgs:
        return False

    last_msg = user_msgs[-1].get("content", "")

    # 条件1：线索词检测
    for cue in SELF_REFLECTION_CUES:
        if cue in last_msg:
            return True

    # 条件2：轮次间隔
    if force_every_n_turns > 0:
        total_user_turns = len(user_msgs)
        # 粗略估算：每 force_every_n_turns 轮用户发言写一条
        expected = total_user_turns // force_every_n_turns
        if agent_self_count < expected:
            return True

    return False


# ============================================================
# 4. 辅助：从历史构建当前场域推断
# ============================================================
def infer_field(history: list, current_field: Optional[str] = None) -> Optional[str]:
    """从最近对话历史推断当前场域。

    简化策略：取最近 3 条用户消息，提取最长有意义短语作为场域名。
    实际使用中应该用 LLM 做 field 分类。

    Args:
        history: 对话历史
        current_field: 如果已知当前场域，直接返回

    Returns:
        场域名称或 None
    """
    if current_field:
        return current_field

    user_msgs = [h["content"] for h in history if h.get("role") == "user"]
    if not user_msgs:
        return None

    # 简化：取最后一条消息的关键实体作为场域
    last = user_msgs[-1]
    # 尝试匹配 "关于...", "...系统", "...框架" 等模式
    m = re.search(r"(?:关于|用|选|做|搭建|开发|重构)([\u4e00-\u9fff\w]{2,8})", last)
    if m:
        return m.group(1)

    return None


# ============================================================
# 5. Hermes 模拟对话脚本
# ============================================================
def simulate_hermes_dialogue():
    """5 轮 Hermes 风格对话模拟，验证 DyadCore 写入与召回行为。"""
    import os
    import time
    from dyadcore import DyadCore

    DB = "hermes_sim.db"
    if os.path.exists(DB):
        os.remove(DB)

    mh = DyadCore(DB)
    print("=" * 60)
    print("HermesBridge 模拟对话 (5 轮) — Dual Mirror")
    print("=" * 60)

    history: list[dict] = []
    agent_self_count = 0
    current_field: Optional[str] = None

    # 预定义 5 轮对话
    rounds = [
        {
            "user": "我想搭建一个本地优先的个人知识库，不用任何云服务",
            "agent": "明白，本地优先是很好的隐私策略。你倾向于用现成工具还是自己拼装？",
            "field": "知识库选型",
            "check_self": True,   # "偏好"线索词
            "check_recall": True, # 冷场触发（首轮）
        },
        {
            "user": "自己拼装，我不喜欢太重的框架。sqlite 做存储你觉得够用吗？",
            "agent": "sqlite 单文件、零配置、性能稳定，对于个人知识库完全够用。FTS5 就能做全文搜索。",
            "field": "知识库选型",
            "check_self": True,   # "不喜欢"线索词
            "check_recall": False, # 同场继续，无触发
        },
        {
            "user": "前端方面，我之前一直用 React，但最近觉得 Vue 3 的 Composition API 更直观",
            "agent": "React 和 Vue 各有优势。Vue 3 的 setup 语法确实更贴近原生 JS 直觉。",
            "field": "前端重构",  # 场域切换！
            "check_self": False,  # 无强线索词
            "check_recall": True, # 名词漂移触发（React→Vue 新话题）
        },
        {
            "user": "话说回来，刚才聊的知识库，你推荐用什么做全文搜索？",
            "agent": "sqlite 内置的 FTS5 就很适合，轻量且够用。",
            "field": "知识库选型",  # 回溯到之前的场
            "check_self": False,
            "check_recall": True, # "刚才聊的" 回溯词触发
        },
        {
            "user": "综合你的建议，我决定用 sqlite + FTS5，前端用 Vue 3。这个组合长期维护应该没问题",
            "agent": "这个技术栈选择扎实：单文件数据库 + 内置全文搜索，全部零依赖。Vue 3 生态也足够稳定。",
            "field": "技术决策",
            "check_self": True,   # "我决定"、"长期" 线索词
            "check_recall": True, # 名词漂移（"综合建议" 新语境）
        },
    ]

    for idx, rnd in enumerate(rounds, 1):
        print(f"\n--- 第 {idx} 轮 ---")
        user_msg = rnd["user"]
        agent_msg = rnd["agent"]
        field = rnd["field"]

        # --- 检测是否触发 recall（用加入当前消息之前的历史）---
        do_recall = should_recall(history)
        trigger_reason = ""
        if do_recall:
            if not history:
                trigger_reason = "（冷场触发）"
            elif any(kw in user_msg for kw in RETROSPECTIVE_KEYWORDS):
                trigger_reason = "（回溯词触发）"
            else:
                trigger_reason = "（名词漂移触发）"

        print(f"  User: {user_msg}")
        print(f"  should_recall: {do_recall} {trigger_reason}")

        # --- 执行 recall ---
        recalled_context = ""
        if do_recall:
            results = mh.recall(query=user_msg, field_hint=field, limit=5)
            recalled_context = format_for_prompt(results, dyadcore=mh)
            print(f"  Recall: {len(results)} 条")
            for i, r in enumerate(results, 1):
                depth_tag = f" [depth={r.get('depth',0)}]" if r.get('depth', 0) > 0 else ""
                print(f"    {i}. [{r['source']} | {r.get('field','')}{depth_tag}] {r['content'][:50]}...")

        # 加入当前用户消息到历史
        history.append({"role": "user", "content": user_msg})

        # --- 写入 user 痕迹 ---
        uid = mh.write(user_msg, memory_type="utterance", source="user", field=field)
        print(f"  写入 user 记忆 id={uid}, field='{field}'")

        # --- 写入 agent 痕迹 ---
        aid = mh.write(agent_msg, memory_type="utterance", source="agent", field=field)
        print(f"  写入 agent 记忆 id={aid}, field='{field}'")

        # --- 检测是否写 agent 观察 ---
        do_self = should_write_agent_self(history, agent_self_count)
        print(f"  should_write_agent_self: {do_self} (已写 {agent_self_count}/{MAX_AGENT_SELF_PER_SESSION})")

        if do_self:
            # 从历史中提取关键偏好，以 agent 视角写入
            user_texts = " ".join(h["content"] for h in history if h["role"] == "user")
            reflection = f"用户在持续表达对本地优先、零依赖、轻量方案的偏好。当前场域: {field}。"
            sid = mh.write(reflection, memory_type="utterance", source="agent", field=field)
            agent_self_count += 1
            print(f"  写入 agent 观察 id={sid}")

        # 加入 agent 消息到历史
        history.append({"role": "agent", "content": agent_msg})

        # 更新当前场域
        current_field = field

    # ========== 最终报告 ==========
    print("\n" + "=" * 60)
    print("模拟结束 — 最终统计 (Dual Mirror)")
    print("=" * 60)

    stats = mh.stats()
    print(f"  总记忆数: {stats['total']}")
    print(f"  锚定: {stats['anchored']}")
    print(f"  活跃: {stats['active']}")
    print(f"  用户痕迹: {stats['user_memories']}, Agent痕迹: {stats['agent_memories']}")
    print(f"  关系边: {stats['total_reflections']}")
    print(f"  场域数: {stats['fields']}")

    print("\n场域分布:")
    for f in mh.list_fields():
        print(f"  {f['field'] or '(无场)'}: {f['count']}条 (锚定{f['anchored_count']}, 用户{f['user_count']}, Agent{f['agent_count']})")

    # 场快照
    print("\n场快照:")
    for f_name in ["知识库选型", "前端重构", "技术决策"]:
        snap = mh.field_snapshot(f_name)
        if snap['total_memories'] > 0:
            print(f"  {f_name}: 强度={snap['strength']:.1f}, "
                  f"极性={snap['polarity']:.2f}, "
                  f"最近关系={snap['recent_reflections']}")

    print("\n--- 验证召回能力 ---")
    results = mh.recall(query="技术决策", field_hint="技术决策", limit=5)
    print(f"  以 '技术决策' 为 field_hint 召回 {len(results)} 条:")
    for i, r in enumerate(results, 1):
        depth_tag = f" [depth={r.get('depth',0)}]" if r.get('depth', 0) > 0 else ""
        print(f"  {i}. [{r['source']} | {r.get('field','')}{depth_tag}] {r['content'][:60]}...")

    print("\n--- 验证回溯词场景召回 ---")
    history2 = [{"role": "user", "content": "回到之前的知识库话题，FTS5 和 BM25 哪个更好？"}]
    if should_recall(history2):
        results = mh.recall(query="FTS5 BM25", field_hint="知识库选型", limit=5)
        print(f"  '回到之前的知识库' 触发召回，返回 {len(results)} 条")
        for i, r in enumerate(results, 1):
            depth_tag = f" [depth={r.get('depth',0)}]" if r.get('depth', 0) > 0 else ""
            print(f"  {i}. [{r['source']} | {r.get('field','')}{depth_tag}] {r['content'][:60]}...")

    # 查看关系图
    print("\n--- 关系图示例 (最新记忆) ---")
    latest = mh.conn.execute(
        "SELECT id FROM memories ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if latest:
        graph = mh.get_relation_graph(latest[0], max_depth=1)
        print(f"  id={latest[0]} 的 1-hop 关系 ({len(graph)} 条边):")
        for edge in graph[:5]:
            print(f"    [{edge['relation_type']}] strength={edge['strength']:.2f} "
                  f"'{edge.get('source_content', '')[:30]}' <-> '{edge.get('target_content', '')[:30]}'")

    mh.close()
    print("\n[OK] HermesBridge 模拟完成")


if __name__ == "__main__":
    simulate_hermes_dialogue()
