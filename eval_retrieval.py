#!/usr/bin/env python3
"""
DyadCore 检索质量基准 (Retrieval Quality Benchmark)
零 LLM 依赖，纯本地评估。

测量指标：
  - Recall@K, Precision@K (K=1,3,5)
  - MRR (Mean Reciprocal Rank)
  - NDCG@K
  - Hit Rate (目标记忆是否出现在 top-K 中)

Ablation 对比：
  - 有/无 field_hint
  - 有/无 anchor
"""

import sys
import os
import random
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dyadcore import DyadCore

# ============================================================================
# 数据集：20 条目标记忆，5 个 field
# ============================================================================

TARGET_MEMORIES = [
    # === personal_info (4) ===
    {"content": "用户正在学习 Rust 编程语言，目前在看 The Rust Book 的第十二章", "field": "personal_info", "type": "utterance", "source": "user"},
    {"content": "用户的猫名叫糯米，是一只两岁的橘猫，非常爱吃鸡胸肉", "field": "personal_info", "type": "utterance", "source": "user"},
    {"content": "用户 prefer 异步沟通，不喜欢突然的会议邀请，希望提前一天通知", "field": "personal_info", "type": "utterance", "source": "user"},
    {"content": "用户每天早晨 7 点到 9 点是深度工作时段，不回复消息", "field": "personal_info", "type": "utterance", "source": "user"},
    # === project_tech (5) ===
    {"content": "团队后端使用 PostgreSQL 15，明确不用 MySQL，因为需要 JSONB 和全文检索功能", "field": "project_tech", "type": "utterance", "source": "user"},
    {"content": "前端框架正在从 Vue 2 迁移到 Vue 3 + TypeScript，预计下月底完成", "field": "project_tech", "type": "utterance", "source": "user"},
    {"content": "CI/CD 使用 GitHub Actions，部署在 AWS ECS Fargate 上", "field": "project_tech", "type": "utterance", "source": "user"},
    {"content": "项目用了 Redis 做缓存层，主要缓存用户的 session 和热点数据", "field": "project_tech", "type": "utterance", "source": "user"},
    {"content": "API 网关用的是 Kong，认证走 OAuth2 + JWT", "field": "project_tech", "type": "utterance", "source": "user"},
    # === meeting_notes (4) ===
    {"content": "上周五和产品经理讨论确定了搜索功能需求：支持模糊搜索、搜索历史、热门搜索推荐", "field": "meeting_notes", "type": "utterance", "source": "user"},
    {"content": "3月15日的迭代评审会上，CTO 强调要优先修复性能问题再上新功能", "field": "meeting_notes", "type": "utterance", "source": "user"},
    {"content": "和设计师达成一致：按钮圆角从 4px 改为 8px，主色调保持 #3B82F6", "field": "meeting_notes", "type": "utterance", "source": "user"},
    {"content": "下周二的 sprint planning 要讨论用户权限系统的重构方案", "field": "meeting_notes", "type": "utterance", "source": "user"},
    # === bugs_issues (3) ===
    {"content": "登录页在 iOS Safari 17.4 上出现无限重定向 bug，已提 issue #342", "field": "bugs_issues", "type": "utterance", "source": "user"},
    {"content": "数据库迁移脚本 D-042 在生产环境执行失败，原因是锁表超时", "field": "bugs_issues", "type": "utterance", "source": "user"},
    {"content": "暗黑模式下的表格边框颜色太浅，用户反馈看不清，优先级 P1", "field": "bugs_issues", "type": "utterance", "source": "user"},
    # === random_thoughts (4) ===
    {"content": "用户觉得 Rust 的所有权系统一开始很难理解，但越用越觉得设计精妙", "field": "random_thoughts", "type": "utterance", "source": "user"},
    {"content": "用户认为 996 工作文化是低效的，好的代码需要清晰的思维而非长时间工作", "field": "random_thoughts", "type": "utterance", "source": "user"},
    {"content": "用户对 AI 辅助编程持开放态度，但认为代码审查仍然需要人工把关", "field": "random_thoughts", "type": "utterance", "source": "user"},
    {"content": "用户最近在读《A Philosophy of Software Design》，觉得第三章关于深模块的论述很有启发", "field": "random_thoughts", "type": "utterance", "source": "user"},
]

# 每个目标记忆对应 3-4 个语义等价的查询
# {target_index: [(query_text, field_hint_or_None), ...]}
QUERIES = {
    0: [
        ("我在学什么编程语言？", "personal_info"),
        ("最近在学什么新东西？", "personal_info"),
        ("我目前的技术学习方向", None),
        ("Rust 学习进度", None),
    ],
    1: [
        ("我的猫叫什么名字？", "personal_info"),
        ("我家宠物是什么品种？", "personal_info"),
        ("糯米是谁？", None),
    ],
    2: [
        ("我喜欢什么样的沟通方式？", "personal_info"),
        ("我对开会有什么偏好吗？", "personal_info"),
        ("怎么跟我协作效率最高？", None),
    ],
    3: [
        ("我什么时候不回复消息？", "personal_info"),
        ("我的深度工作时段是几点？", "personal_info"),
        ("早上能找到我吗？", None),
    ],
    4: [
        ("我们用什么数据库？", "project_tech"),
        ("后端数据库选型", "project_tech"),
        ("为什么不用 MySQL？", None),
        ("PostgreSQL 版本是多少？", None),
    ],
    5: [
        ("前端现在用什么框架？", "project_tech"),
        ("Vue 迁移的进度如何？", "project_tech"),
        ("前端技术栈是什么？", None),
    ],
    6: [
        ("我们用什么做 CI/CD？", "project_tech"),
        ("部署在哪里？", "project_tech"),
        ("服务怎么部署的？", None),
    ],
    7: [
        ("Redis 用来做什么？", "project_tech"),
        ("缓存层用了什么技术？", "project_tech"),
        ("session 数据存在哪里？", None),
    ],
    8: [
        ("API 网关是什么？", "project_tech"),
        ("认证机制怎么做的？", "project_tech"),
        ("Kong 和 JWT 的配置", None),
    ],
    9: [
        ("搜索功能的需求是什么？", "meeting_notes"),
        ("产品经理对搜索提了什么要求？", "meeting_notes"),
        ("搜索要做哪些功能？", None),
    ],
    10: [
        ("CTO 在评审会上说了什么？", "meeting_notes"),
        ("性能问题的优先级怎么样？", "meeting_notes"),
        ("先修 bug 还是先上新功能？", None),
    ],
    11: [
        ("按钮圆角的设计决策", "meeting_notes"),
        ("UI 设计规范改了什么？", "meeting_notes"),
        ("和设计师定了什么？", None),
    ],
    12: [
        ("下周有什么会议？", "meeting_notes"),
        ("sprint planning 要讨论什么？", "meeting_notes"),
        ("权限系统什么时候讨论？", None),
    ],
    13: [
        ("iOS Safari 有什么 bug？", "bugs_issues"),
        ("登录页的问题是什么？", "bugs_issues"),
        ("issue #342 是关于什么的？", None),
    ],
    14: [
        ("数据库迁移出了什么问题？", "bugs_issues"),
        ("D-042 脚本怎么了？", "bugs_issues"),
        ("生产环境的数据库问题", None),
    ],
    15: [
        ("暗黑模式有什么问题？", "bugs_issues"),
        ("表格在暗黑模式下怎么样？", "bugs_issues"),
        ("P1 的 UI bug 是什么？", None),
    ],
    16: [
        ("用户怎么评价 Rust 的所有权系统？", "random_thoughts"),
        ("用户对 Rust 有什么看法？", "random_thoughts"),
        ("Rust 难学吗？", None),
    ],
    17: [
        ("用户怎么看 996？", "random_thoughts"),
        ("用户对加班文化的态度", "random_thoughts"),
        ("高效工作的关键是什么？", None),
    ],
    18: [
        ("用户对 AI 编程怎么看？", "random_thoughts"),
        ("对 AI 辅助编程持什么态度？", "random_thoughts"),
        ("代码审查还需要人来做吗？", None),
    ],
    19: [
        ("用户最近在读什么书？", "random_thoughts"),
        ("《A Philosophy of Software Design》怎么样？", "random_thoughts"),
        ("深模块是什么概念？", None),
    ],
}

# ============================================================================
# 噪音填充记忆
# ============================================================================

FILLER_TEMPLATES = [
    "关于{subject}的讨论还在继续，目前没有明确结论",
    "我查了一下{subject}的文档，发现{detail}",
    "{subject}的最佳实践是什么？想听听你的建议",
    "昨天看了{subject}的源码，发现{detail}",
    "有没有{subject}的替代方案？{detail}",
    "{subject}的测试覆盖率目前只有{percent}%，需要提高",
    "关于{subject}，我觉得{opinion}",
    "今天在{subject}上花了{hours}个小时，进度{progress}",
    "需要重构{subject}模块，代码太乱了",
    "{subject}的文档需要更新，{detail}已经过时了",
    "团队里有人提议用{subject}替换现有的方案",
    "{subject}的性能基准测试结果出来了，{detail}",
    "我觉得{subject}的设计模式有问题，{detail}",
    "看了{subject}的 changelog，新版本{detail}",
    "有没有人用过{subject}？想了解一下实际体验",
    "{subject}的配置文件太复杂了，{detail}",
    "给{subject}提了一个 PR，修复了{detail}",
    "{subject}的 API 设计得不错，{detail}",
    "遇到一个{subject}的坑，{detail}",
    "分享一篇关于{subject}的好文章，{detail}",
    "老板问{subject}的进展，我说{detail}",
    "周末研究了一下{subject}，感觉{detail}",
    "{subject}的代码审查完成了，主要问题是{detail}",
    "关于{subject}的决定：我们最终选择了{detail}",
    "今天面试了一个候选人，问他{subject}，回答{detail}",
    "和同事 pair programming 做{subject}，学到了{detail}",
    "{subject}的上线延期了，原因是{detail}",
    "{subject}的监控报警响了，排查发现{detail}",
    "写了{subject}的单元测试，覆盖了{detail}",
    "{subject}的代码合入了 main 分支，{detail}",
]

SUBJECTS = [
    "微服务架构", "Docker 部署", "负载均衡", "数据库索引", "消息队列",
    "日志系统", "监控告警", "权限管理", "文件上传", "数据导出",
    "邮件服务", "短信验证", "第三方登录", "支付集成", "地图服务",
    "推送通知", "数据备份", "灰度发布", "AB测试", "错误追踪",
    "性能优化", "代码生成器", "接口文档", "状态管理", "路由设计",
    "表单校验", "分页组件", "图表可视化", "国际化", "无障碍访问",
]

DETAILS = [
    "配置项太多了，容易出错", "文档写得不够清楚", "性能比预期好", "社区不够活跃",
    "和现有系统的兼容性有问题", "需要额外的中间件支持", "学习曲线陡峭",
    "维护成本比较高", "缺少关键功能", "API 设计得很优雅", "bug 太多了",
    "版本更新太快", "对 TypeScript 支持不完整", "在生产环境表现稳定",
    "和 PostgreSQL 的集成很方便", "内存占用有点高", "冷启动时间太长",
    "并发处理能力很强", "错误信息不够友好", "扩展性很好",
]

OPINIONS = [
    "应该优先考虑可维护性", "过度设计反而不好", "简单方案更可靠",
    "先跑起来再优化", "自动化测试不能省", "文档比代码更重要",
    "团队统一技术栈很重要", "不要过早优化", "代码可读性第一",
    "约定优于配置", "能不用第三方库就不用",
]

PROGRESSES = ["完成了 30%", "卡住了", "还算顺利", "基本搞定", "刚开始做", "需要返工"]
PERCENTS = ["30", "45", "60", "75", "80", "90"]
HOURS = ["2", "3", "4", "1.5", "半天", "一整天"]


def generate_fillers(count=180):
    """生成随机噪音记忆。"""
    fillers = []
    fields = ["项目讨论", "技术调研", "日常沟通", "问题排查", "代码审查"]
    for _ in range(count):
        tmpl = random.choice(FILLER_TEMPLATES)
        content = tmpl.format(
            subject=random.choice(SUBJECTS),
            detail=random.choice(DETAILS),
            opinion=random.choice(OPINIONS),
            progress=random.choice(PROGRESSES),
            percent=random.choice(PERCENTS),
            hours=random.choice(HOURS),
        )
        fillers.append({
            "content": content,
            "field": random.choice(fields),
            "type": "utterance",
            "source": random.choice(["user", "agent"]),
        })
    return fillers


# ============================================================================
# 评估指标
# ============================================================================

def recall_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(relevant_ids) & set(retrieved_ids[:k])) / len(relevant_ids)


def precision_at_k(relevant_ids, retrieved_ids, k):
    if k == 0:
        return 0.0
    return len(set(relevant_ids) & set(retrieved_ids[:k])) / k


def mrr(relevant_ids, retrieved_ids):
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant_ids, retrieved_ids, k):
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# 主评估
# ============================================================================

def main():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_retrieval.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=" * 72)
    print("  DyadCore 检索质量基准")
    print("=" * 72)

    mh = DyadCore(db_path)
    print(f"\n  检索引擎: FTS5 trigram + LIKE fallback + 关系图扩展 (Dual Mirror, zero-dependency)")

    # ==================================================================
    # Phase 1: 写入记忆
    # ==================================================================
    print("\n" + "-" * 72)
    print("  Phase 1: 写入记忆")
    print("-" * 72)

    target_ids = {}
    for i, mem in enumerate(TARGET_MEMORIES):
        mid = mh.write(
            content=mem["content"],
            memory_type=mem["type"],
            source=mem["source"],
            field=mem["field"],
        )
        target_ids[i] = mid
    print(f"  目标记忆: {len(target_ids)} 条")

    random.seed(42)
    fillers = generate_fillers(180)
    filler_ids = []
    for mem in fillers:
        mid = mh.write(
            content=mem["content"],
            memory_type=mem["type"],
            source=mem["source"],
            field=mem["field"],
        )
        filler_ids.append(mid)
    print(f"  噪音记忆: {len(filler_ids)} 条")
    print(f"  总计:      {len(target_ids) + len(filler_ids)} 条")

    db_size = os.path.getsize(db_path)
    print(f"  DB 大小:   {db_size / 1024:.1f} KB")
    print(f"  各 Field 分布:")
    for f in mh.list_fields():
        print(f"    {f['field'] or '(null)':<22} {f['count']:>4} 条")

    # ==================================================================
    # Phase 2: 基准召回测试（无 anchor，有 field_hint）
    # ==================================================================
    print("\n" + "-" * 72)
    print("  Phase 2: 基准召回测试 (无 Anchor，默认 field_hint)")
    print("-" * 72)

    # 收集所有 query 的详细结果
    query_results = []  # [{target_idx, query, field_hint, hit, rank, retrieved_ids, metrics}]

    for target_idx, query_list in QUERIES.items():
        relevant_ids = {target_ids[target_idx]}
        for q_text, q_field in query_list:
            results = mh.recall(query=q_text, field_hint=q_field, limit=5)
            retrieved_ids = [r["id"] for r in results]

            hit = target_ids[target_idx] in retrieved_ids
            rank = retrieved_ids.index(target_ids[target_idx]) + 1 if hit else None

            query_results.append({
                "target_idx": target_idx,
                "target_field": TARGET_MEMORIES[target_idx]["field"],
                "target_content": TARGET_MEMORIES[target_idx]["content"],
                "query": q_text,
                "field_hint": q_field,
                "hit": hit,
                "rank": rank,
                "retrieved_ids": retrieved_ids,
                "recall@1": recall_at_k(relevant_ids, retrieved_ids, 1),
                "recall@3": recall_at_k(relevant_ids, retrieved_ids, 3),
                "recall@5": recall_at_k(relevant_ids, retrieved_ids, 5),
                "precision@1": precision_at_k(relevant_ids, retrieved_ids, 1),
                "precision@3": precision_at_k(relevant_ids, retrieved_ids, 3),
                "precision@5": precision_at_k(relevant_ids, retrieved_ids, 5),
                "mrr": mrr(relevant_ids, retrieved_ids),
                "ndcg@5": ndcg_at_k(relevant_ids, retrieved_ids, 5),
            })

    # 汇总指标
    def summarize(qrs):
        keys = ["recall@1", "recall@3", "recall@5",
                "precision@1", "precision@3", "precision@5",
                "mrr", "ndcg@5"]
        return {k: sum(r[k] for r in qrs) / len(qrs) for k in keys}

    summary = summarize(query_results)

    print(f"\n  查询总数: {len(query_results)}")
    print(f"\n  {'指标':<20} {'均值':>8}  {'解读'}")
    print(f"  {'-'*52}")
    metric_notes = {
        "recall@1": "目标在第一位",
        "recall@3": "目标在前三位",
        "recall@5": "目标在前五位",
        "precision@1": "第一位是目标的概率",
        "precision@3": "前三位中目标的占比",
        "precision@5": "前五位中目标的占比",
        "mrr": "首个命中排名的倒数均值",
        "ndcg@5": "归一化折损累计增益",
    }
    for k in ["recall@1", "recall@3", "recall@5",
              "precision@1", "precision@3", "precision@5",
              "mrr", "ndcg@5"]:
        print(f"  {k:<20} {summary[k]:>8.4f}  {metric_notes[k]}")

    hit_at = {}
    for k in [1, 3, 5]:
        hit_at[k] = sum(1 for r in query_results
                        if r["rank"] and r["rank"] <= k) / len(query_results)

    print(f"\n  {'Hit Rate':<20} {'值':>8}")
    print(f"  {'-'*28}")
    for k in [1, 3, 5]:
        print(f"  Hit@{k:<17} {hit_at[k]:>8.2%}")

    # ==================================================================
    # Phase 2.5: 诊断 —— 目标记忆在向量搜索 Top-100 中的位置
    # ==================================================================
    print("\n" + "-" * 72)
    print("  Phase 2.5: FTS5 查询覆盖率诊断")
    print("-" * 72)

    fts_stats = {"hit": 0, "miss": 0, "by_field": defaultdict(lambda: {"hit": 0, "miss": 0})}

    for target_idx, query_list in QUERIES.items():
        target_mid = target_ids[target_idx]
        target_field = TARGET_MEMORIES[target_idx]["field"]
        for q_text, q_field in query_list:
            fts_query = mh._build_fts5_query(q_text)
            # 直接 FTS5 查询：目标记忆是否在 top-100 中
            top100 = mh.conn.execute("""
                SELECT rowid AS memory_id, rank
                FROM memory_fts
                WHERE content MATCH ?
                ORDER BY rank
                LIMIT 100
            """, (fts_query,)).fetchall()

            fts_rank = None
            for i, r in enumerate(top100, 1):
                if r[0] == target_mid:
                    fts_rank = i
                    break

            if fts_rank:
                fts_stats["hit"] += 1
                fts_stats["by_field"][target_field]["hit"] += 1
            else:
                fts_stats["miss"] += 1
                fts_stats["by_field"][target_field]["miss"] += 1

    total_q = fts_stats["hit"] + fts_stats["miss"]
    fts100_rate = fts_stats["hit"] / total_q if total_q > 0 else 0

    print(f"\n  目标记忆进入 FTS5 Top-100 的比例: {fts100_rate:.2%} ({fts_stats['hit']}/{total_q})")
    print(f"\n  按 Field 分组:")
    for field in sorted(fts_stats["by_field"]):
        fs = fts_stats["by_field"][field]
        total = fs["hit"] + fs["miss"]
        rate = fs["hit"] / total if total > 0 else 0
        print(f"    {field:<22} FTS5命中率: {fs['hit']}/{total} ({rate:.0%})")

    print(f"\n  [!] 诊断结论:")
    if fts100_rate < 0.7:
        print(f"  FTS5 检索 Top-100 命中率仅 {fts100_rate:.0%}，部分查询构造可能不够理想")
        print(f"  建议: 针对 CJK 查询优化 _build_fts5_query 的滑动窗口策略")
    else:
        print(f"  FTS5 检索工作正常。如果 Hit Rate 仍低，问题在排序/关系图扩展阶段。")

    # ==================================================================
    # Phase 3: Ablation Study
    # ==================================================================
    print("\n" + "-" * 72)
    print("  Phase 3: Ablation Study")
    print("-" * 72)

    ablation_configs = [
        ("A. 基准 (有 field_hint, 无 anchor)",      True,  False),
        ("B. 无 field_hint, 无 anchor",              False, False),
    ]

    ablation_results = {}

    # Run non-anchor configs first
    for config_name, use_fh, use_anchor in ablation_configs:
        agg = {"recall@5": [], "mrr": [], "ndcg@5": [], "hit": []}

        for target_idx, query_list in QUERIES.items():
            relevant_ids = {target_ids[target_idx]}
            for q_text, q_field in query_list:
                fh = q_field if use_fh else None
                results = mh.recall(query=q_text, field_hint=fh, limit=5)
                retrieved_ids = [r["id"] for r in results]

                agg["recall@5"].append(recall_at_k(relevant_ids, retrieved_ids, 5))
                agg["mrr"].append(mrr(relevant_ids, retrieved_ids))
                agg["ndcg@5"].append(ndcg_at_k(relevant_ids, retrieved_ids, 5))
                agg["hit"].append(1.0 if target_ids[target_idx] in retrieved_ids else 0.0)

        ablation_results[config_name] = {k: sum(v) / len(v) for k, v in agg.items()}

    # Now anchor all targets
    for idx in target_ids.values():
        mh.anchor(idx)

    # Run anchor configs
    anchor_configs = [
        ("C. 有 field_hint + Anchor",  True,  True),
        ("D. 无 field_hint + Anchor",  False, True),
    ]

    for config_name, use_fh, use_anchor in anchor_configs:
        agg = {"recall@5": [], "mrr": [], "ndcg@5": [], "hit": []}

        for target_idx, query_list in QUERIES.items():
            relevant_ids = {target_ids[target_idx]}
            for q_text, q_field in query_list:
                fh = q_field if use_fh else None
                results = mh.recall(query=q_text, field_hint=fh, limit=5)
                retrieved_ids = [r["id"] for r in results]

                agg["recall@5"].append(recall_at_k(relevant_ids, retrieved_ids, 5))
                agg["mrr"].append(mrr(relevant_ids, retrieved_ids))
                agg["ndcg@5"].append(ndcg_at_k(relevant_ids, retrieved_ids, 5))
                agg["hit"].append(1.0 if target_ids[target_idx] in retrieved_ids else 0.0)

        ablation_results[config_name] = {k: sum(v) / len(v) for k, v in agg.items()}

    # Print ablation comparison table
    header = f"\n  {'Config':<38} {'Recall@5':>9} {'MRR':>9} {'NDCG@5':>9} {'Hit':>9}"
    print(header)
    print(f"  {'-'*76}")
    best = {k: max(ablation_results[c][k] for c in ablation_results)
            for k in ["recall@5", "mrr", "ndcg@5", "hit"]}
    for config_name in ablation_results:
        r = ablation_results[config_name]
        markers = []
        for k in ["recall@5", "mrr", "ndcg@5", "hit"]:
            markers.append(" *" if r[k] == best[k] else "  ")
        print(f"  {config_name:<38} {r['recall@5']:>8.4f}{markers[0]} "
              f"{r['mrr']:>8.4f}{markers[1]} "
              f"{r['ndcg@5']:>8.4f}{markers[2]} "
              f"{r['hit']:>7.2%}{markers[3]}")
    print(f"\n  (* = 该列最优)")

    # Anchor boost calculation
    base = ablation_results["A. 基准 (有 field_hint, 无 anchor)"]
    anchored = ablation_results["C. 有 field_hint + Anchor"]
    print(f"\n  Anchor Boost (C vs A):")
    for k in ["recall@5", "mrr", "ndcg@5", "hit"]:
        delta = anchored[k] - base[k]
        pct = (delta / base[k] * 100) if base[k] > 0 else 0
        print(f"    {k:<15} {base[k]:.4f} -> {anchored[k]:.4f}  (Δ={delta:+.4f}, {pct:+.1f}%)")

    # ==================================================================
    # Phase 4: 精细分析
    # ==================================================================
    print("\n" + "-" * 72)
    print("  Phase 4: 精细分析")
    print("-" * 72)

    # 4a: 按 field 分组
    print("\n  [4a] 按 Field 分组的 Hit Rate:")
    field_stats = defaultdict(lambda: {"hits": 0, "total": 0, "ranks": []})
    for r in query_results:
        f = r["target_field"]
        field_stats[f]["total"] += 1
        if r["hit"]:
            field_stats[f]["hits"] += 1
            field_stats[f]["ranks"].append(r["rank"])

    for field in sorted(field_stats):
        s = field_stats[field]
        rate = s["hits"] / s["total"]
        avg_rank = sum(s["ranks"]) / len(s["ranks"]) if s["ranks"] else float("inf")
        print(f"    {field:<22} Hit:{rate:>7.2%} ({s['hits']}/{s['total']})  "
              f"AvgRank:{avg_rank:.1f}")

    # 4b: field_hint 效果
    print("\n  [4b] Field Hint 效果:")
    with_hint = [r for r in query_results if r["field_hint"] is not None]
    without_hint = [r for r in query_results if r["field_hint"] is None]
    for label, subset in [("有 field_hint", with_hint), ("无 field_hint", without_hint)]:
        hits = sum(1 for r in subset if r["hit"])
        avg_mrr = sum(r["mrr"] for r in subset) / len(subset)
        print(f"    {label}:  Hit={hits}/{len(subset)} ({hits/len(subset):.2%})  MRR={avg_mrr:.4f}")

    # 4c: 正确 field_hint vs 错误 field_hint
    print("\n  [4c] Field Hint 准确性影响:")
    correct_hint = [r for r in query_results
                    if r["field_hint"] is not None and r["field_hint"] == r["target_field"]]
    wrong_hint = [r for r in query_results
                  if r["field_hint"] is not None and r["field_hint"] != r["target_field"]]
    for label, subset in [("正确 field_hint", correct_hint), ("跨 field_hint", wrong_hint)]:
        if subset:
            hits = sum(1 for r in subset if r["hit"])
            avg_mrr = sum(r["mrr"] for r in subset) / len(subset)
            print(f"    {label}:  Hit={hits}/{len(subset)} ({hits/len(subset):.2%})  MRR={avg_mrr:.4f}")
        else:
            print(f"    {label}:  (无数据)")

    # 4d: 所有未命中的查询
    print("\n  [4d] 未命中的查询:")
    misses = [r for r in query_results if not r["hit"]]
    if misses:
        for i, r in enumerate(misses):
            print(f"\n    #{i+1} Query: \"{r['query']}\"")
            print(f"       Field Hint: {r['field_hint']}  |  Target Field: {r['target_field']}")
            print(f"       Target: \"{r['target_content'][:100]}...\"")
    else:
        print("    全部命中！")

    # 4e: MRR 分布直方图
    print("\n  [4e] MRR 分布:")
    mrr_buckets = {
        "MRR = 1.000 (首位)": 0,
        "0.500 ≤ MRR < 1.000": 0,
        "0.333 ≤ MRR < 0.500": 0,
        "0.200 ≤ MRR < 0.333": 0,
        "0.000 < MRR < 0.200": 0,
        "MRR = 0.000 (未命中)": 0,
    }
    for r in query_results:
        v = r["mrr"]
        if v == 1.0:
            mrr_buckets["MRR = 1.000 (首位)"] += 1
        elif v >= 0.5:
            mrr_buckets["0.500 ≤ MRR < 1.000"] += 1
        elif v >= 0.333:
            mrr_buckets["0.333 ≤ MRR < 0.500"] += 1
        elif v >= 0.2:
            mrr_buckets["0.200 ≤ MRR < 0.333"] += 1
        elif v > 0:
            mrr_buckets["0.000 < MRR < 0.200"] += 1
        else:
            mrr_buckets["MRR = 0.000 (未命中)"] += 1

    for label, count in mrr_buckets.items():
        pct = count / len(query_results) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<30} {count:>3} ({pct:>5.1f}%) {bar}")

    # 4f: Query 类型分析
    print("\n  [4f] 查询表述类型分析:")
    # 将查询分为：精确匹配、语义相近、跨域引用
    exact_style = [r for r in query_results
                   if any(kw in r["query"] for kw in ["什么", "哪个", "多少", "几", "谁"])]
    semantic_style = [r for r in query_results if r not in exact_style]
    for label, subset in [("疑问词查询 (什么/哪个/谁)", exact_style),
                           ("语义查询 (描述性)", semantic_style)]:
        hits = sum(1 for r in subset if r["hit"])
        avg_mrr = sum(r["mrr"] for r in subset) / len(subset)
        print(f"    {label}:  Hit={hits}/{len(subset)} ({hits/len(subset):.2%})  MRR={avg_mrr:.4f}")

    # ==================================================================
    # 总结
    # ==================================================================
    print("\n" + "=" * 72)
    print("  总结")
    print("=" * 72)

    overall_hit = sum(1 for r in query_results if r["hit"]) / len(query_results)
    overall_mrr = summary["mrr"]
    overall_r5 = summary["recall@5"]

    # Grade
    if overall_mrr >= 0.8:
        grade = "A — 检索质量优秀"
    elif overall_mrr >= 0.6:
        grade = "B — 检索质量良好，有优化空间"
    elif overall_mrr >= 0.4:
        grade = "C — 检索质量一般，建议排查"
    else:
        grade = "D — 检索质量差，需要检查 Embedding 和排序逻辑"

    print(f"""
  整体 Hit Rate:  {overall_hit:.2%}
  整体 MRR:       {overall_mrr:.4f}
  整体 Recall@5:  {overall_r5:.4f}
  检索引擎: FTS5 trigram + LIKE + Graph (Dual Mirror)

  评级: {grade}

  未命中查询: {len(misses)}/{len(query_results)}
  数据库:     {db_path}
    """)

    mh.close()
    # 保留 db 文件供人工检查
    print(f"  数据库已保留: {db_path} (可供人工检查)\n")


if __name__ == "__main__":
    main()
