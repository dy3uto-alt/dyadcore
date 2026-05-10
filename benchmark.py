#!/usr/bin/env python3
"""
DyadCore 本地性能基准测试 — 纯 SQLite 操作，零 LLM API 消耗。

测试项：
  1. 批量写入 1000 条随机记忆
  2. 不同场景下的 recall 延迟
  3. FTS5 全文搜索性能
  4. 管理操作（anchor/archive/stats）
"""

import os
import sys
import time
import random
import math
from dyadcore import DyadCore

# ============================================================
# 随机内容池 — 模拟真实对话中的多类型记忆
# ============================================================
UTTERANCE_POOL = [
    "我想用 React 19 的 Server Components 做 SSR",
    "sqlite 的单文件架构非常适合本地优先应用",
    "你觉得 TypeScript 的 strict 模式有必要开吗",
    "Vue 3 的 Composition API 比 Options API 灵活很多",
    "PostgreSQL 的全文搜索比 MySQL 强太多了",
    "Docker 在本地开发中还是太重了，换 podman 吧",
    "Rust 的内存安全模型确实让人放心",
    "Python 的类型提示越来越完善了",
    "我们上次讨论的那个方案还需要再优化一下",
    "Mozilla 的 sqlite-vec 扩展支持向量检索",
    "Tailwind CSS 比 BEM 命名法更直观",
    "Prisma ORM 对 SQLite 的支持也很完善",
    "Electron 打包太大了，Tauri 更轻量",
    "GitHub Actions 的 CI/CD 配置比 Jenkins 简单",
    "WebAssembly 在浏览器端的性能接近原生",
    "GraphQL 的 N+1 问题可以用 DataLoader 解决",
    "Redis 做缓存层比分页查询快一个数量级",
    "Kubernetes 对于小团队来说维护成本太高",
    "Svelte 的编译时方案比运行时虚拟 DOM 更高效",
    "JWT 的无状态特性减少了服务端存储压力",
    "Markdown 写作体验比富文本编辑器更专注",
    "Vim 的编辑效率在熟练后远超其他编辑器",
    "Deno 2.0 的原生 TypeScript 支持非常方便",
    "Astro 的岛屿架构对内容站非常友好",
    "Elixir 的 OTP 框架在并发场景下表现优异",
    "Zig 作为 C 的替代品在系统编程中很有潜力",
    "htmx 让前端交互回归了服务端渲染的简洁",
    "Kafka 的事件溯源模式适合金融系统",
    "Obsidian 的本地 Markdown 知识库体验很好",
    "Linux 的 systemd 虽然争议大但确实统一了服务管理",
]

ACTION_POOL = [
    "执行了数据库迁移 v2.3",
    "部署了新版本到 staging 环境",
    "创建了 PR #42 修复内存泄漏",
    "合并了 feature/auth-refactor 分支",
    "运行了完整的集成测试套件",
    "更新了依赖包到最新版本",
    "添加了 3 个新的 API 端点",
    "重构了 UserService 模块",
    "删除了废弃的旧接口",
    "发布了 v1.2.0 到生产环境",
    "配置了 Sentry 错误监控",
    "添加了 Redis 缓存层",
    "修复了 CORS 跨域问题",
    "优化了慢查询的性能",
    "启用了 CDN 加速静态资源",
    "更新了 ESLint 规则配置",
    "迁移了 CI 从 CircleCI 到 GitHub Actions",
    "为 API 添加了 rate limiting",
    "重构了前端路由结构",
    "实现了用户权限的 RBAC 模型",
]

META_POOL = [
    "项目总代码行数: 12,400",
    "当前活跃分支: feature/memory-system",
    "测试覆盖率: 73.2% (target: 80%)",
    "上次部署: 2026-04-29 14:30 UTC",
    "依赖库数量: 23 (全部通过 audit)",
    "API 可用率: 99.97% (本月)",
    "P95 响应时间: 320ms",
    "未解决 issue 数: 17",
    "当前迭代还剩 3 个工作日",
    "团队规模: 4 人全职 + 1 人兼职",
]

FIELD_POOL = [
    "技术选型", "前端重构", "后端优化", "DevOps",
    "数据库设计", "API 设计", "测试策略", "安全审计",
    "性能优化", "代码审查", None, None, None,  # 30% 无场域
]

SOURCE_POOL = ["user", "agent"]


def random_text(pool, min_words=5, max_words=20):
    """从池中随机选取并拼接生成可变长度文本。"""
    text = random.choice(pool)
    # 30% 概率追加额外内容
    if random.random() < 0.3:
        extra = random.choice(pool)
        text += "，此外" + extra
    return text


def format_ms(seconds):
    """秒转可读格式。"""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.3f} s"


def stats_summary(samples):
    """计算百分位统计。"""
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0],
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
        "mean": sum(s) / n,
        "std": (sum((x - sum(s) / n) ** 2 for x in s) / n) ** 0.5,
    }


def main():
    DB = "benchmark.db"
    if os.path.exists(DB):
        os.remove(DB)

    mh = DyadCore(DB)

    tier_names = {1: "Ollama", 2: "Sentence-Transformers", 3: "FTS5-only"}
    print("=" * 64)
    print("  DyadCore 本地性能基准测试 (Dual Mirror)")
    print(f"  检索引擎: FTS5 trigram + Graph")
    print(f"  DB Path: {os.path.abspath(DB)}")
    print("=" * 64)

    # ============================================================
    # Phase 1: 批量写入 1000 条
    # ============================================================
    TOTAL = 1000
    BATCH = 100
    print(f"\n{'='*64}")
    print(f"  Phase 1: 写入 {TOTAL} 条随机记忆")
    print(f"{'='*64}")

    write_times = []
    type_dist = {"utterance": 0, "action": 0, "meta": 0}
    field_dist = {}
    anchored_count = 0

    t_start = time.perf_counter()

    for i in range(TOTAL):
        t_write = time.perf_counter()

        # 混合类型: 70% utterance, 20% action, 10% meta
        r = random.random()
        if r < 0.70:
            mtype = "utterance"
            content = random_text(UTTERANCE_POOL)
        elif r < 0.90:
            mtype = "action"
            content = random_text(ACTION_POOL)
        else:
            mtype = "meta"
            content = random_text(META_POOL)

        source = random.choice(SOURCE_POOL)
        field = random.choice(FIELD_POOL)

        mid = mh.write(content, memory_type=mtype, source=source, field=field)

        # 约 5% 的记忆锚定
        if random.random() < 0.05:
            mh.anchor(mid)
            anchored_count += 1

        elapsed = time.perf_counter() - t_write
        write_times.append(elapsed)

        type_dist[mtype] += 1
        fkey = field or "(无场)"
        field_dist[fkey] = field_dist.get(fkey, 0) + 1

        # 每 100 条输出一次进度
        if (i + 1) % BATCH == 0:
            batch_times = write_times[-BATCH:]
            batch_total = sum(batch_times)
            batch_mean = batch_total / BATCH * 1000
            print(f"  [{i+1:4d}/{TOTAL}] "
                  f"批次 {BATCH} 条: {format_ms(batch_total)} "
                  f"(均 {batch_mean:.1f} ms/条)")

    t_total_write = time.perf_counter() - t_start

    wstat = stats_summary(write_times)

    print(f"\n  --- 写入统计 ---")
    print(f"  总耗时:     {format_ms(t_total_write)}")
    print(f"  总条数:     {TOTAL}")
    print(f"  均值:       {wstat['mean'] * 1000:.2f} ms/条")
    print(f"  P50:        {wstat['p50'] * 1000:.2f} ms")
    print(f"  P95:        {wstat['p95'] * 1000:.2f} ms")
    print(f"  P99:        {wstat['p99'] * 1000:.2f} ms")
    print(f"  吞吐量:     {TOTAL / t_total_write:.1f} 条/秒")
    print(f"  锚定:       {anchored_count}/{TOTAL}")
    print(f"  类型分布:   utterance={type_dist['utterance']}, "
          f"action={type_dist['action']}, meta={type_dist['meta']}")

    # ============================================================
    # Phase 2: Recall 性能测试
    # ============================================================
    print(f"\n{'='*64}")
    print(f"  Phase 2: Recall 延迟测试")
    print(f"{'='*64}")

    recall_scenarios = [
        ("无 query + field_hint", dict(field_hint="技术选型", limit=10)),
        ("有 query + field_hint", dict(query="React 前端框架", field_hint="前端重构", limit=10)),
        ("有 query 无 field_hint", dict(query="数据库性能优化", limit=10)),
        ("无 query 无 field_hint (全锚定)", dict(limit=10)),
        ("大 limit=50 + field_hint", dict(field_hint="API 设计", limit=50)),
        ("大 limit=50 无 field_hint", dict(limit=50)),
        ("仅锚定记忆召回", dict(limit=5)),  # 无 field_hint，靠 anchor 排序
    ]

    for label, kwargs in recall_scenarios:
        samples = []
        warmup = 2
        runs = 10

        for _ in range(warmup + runs):
            t0 = time.perf_counter()
            results = mh.recall(**kwargs)
            samples.append(time.perf_counter() - t0)

        # 丢弃预热
        samples = samples[warmup:]

        rstat = stats_summary(samples)
        # 获取最后一次的结果数量
        final_count = len(results)

        print(f"\n  [{label}]")
        print(f"    返回条数:   {final_count}")
        print(f"    P50 延迟:   {format_ms(rstat['p50'])}")
        print(f"    P95 延迟:   {format_ms(rstat['p95'])}")
        print(f"    均延迟:     {format_ms(rstat['mean'])}")

    # ============================================================
    # Phase 3: FTS5 全文搜索
    # ============================================================
    print(f"\n{'='*64}")
    print(f"  Phase 3: FTS5 全文搜索")
    print(f"{'='*64}")

    fts_queries = [
        "React",
        "SQLite",
        "Docker",
        "Vue",
        "数据库",
        "性能优化",
        "部署",
        "TypeScript",
    ]

    for q in fts_queries:
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            results = mh.search_fts(q, limit=20)
            samples.append(time.perf_counter() - t0)

        fstat = stats_summary(samples)
        print(f"  FTS5 '{q}': {len(results):2d} 条, "
              f"P50={format_ms(fstat['p50'])}, "
              f"均={format_ms(fstat['mean'])}")

    # ============================================================
    # Phase 4: 管理操作
    # ============================================================
    print(f"\n{'='*64}")
    print(f"  Phase 4: 管理操作")
    print(f"{'='*64}")

    # stats
    t0 = time.perf_counter()
    st = mh.stats()
    print(f"  stats():       {format_ms(time.perf_counter() - t0)} -> total={st['total']}")

    # list_fields
    t0 = time.perf_counter()
    fields = mh.list_fields()
    print(f"  list_fields(): {format_ms(time.perf_counter() - t0)} -> {len(fields)} fields")

    # check_silence
    t0 = time.perf_counter()
    silent = mh.check_silence(days=30)
    print(f"  check_silence(30d): {format_ms(time.perf_counter() - t0)} -> {len(silent)} silent")

    # check_silence_by_field
    t0 = time.perf_counter()
    silent_by_field = mh.check_silence_by_field(days=30)
    print(f"  check_silence_by_field(30d): {format_ms(time.perf_counter() - t0)} -> {len(silent_by_field)} groups")

    # get_memory (随机 10 次)
    get_samples = []
    for _ in range(10):
        mid = random.randint(1, TOTAL)
        t0 = time.perf_counter()
        mh.get_memory(mid)
        get_samples.append(time.perf_counter() - t0)
    gstat = stats_summary(get_samples)
    print(f"  get_memory() x10: P50={format_ms(gstat['p50'])}, 均={format_ms(gstat['mean'])}")

    # ============================================================
    # Phase 5: 存储统计
    # ============================================================
    print(f"\n{'='*64}")
    print(f"  Phase 5: 存储统计")
    print(f"{'='*64}")

    count_mem = mh.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    count_ref = mh.conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
    db_size = os.path.getsize(DB)
    wal_size = os.path.getsize(DB + "-wal") if os.path.exists(DB + "-wal") else 0
    shm_size = os.path.getsize(DB + "-shm") if os.path.exists(DB + "-shm") else 0

    print(f"  memories 表行数:          {count_mem}")
    print(f"  reflections 表行数:       {count_ref}")
    print(f"  .db 文件大小:             {db_size / 1024:.1f} KB")
    print(f"  .db-wal 文件大小:         {wal_size / 1024:.1f} KB")
    print(f"  .db-shm 文件大小:         {shm_size / 1024:.1f} KB")
    print(f"  存储总计:                 {(db_size + wal_size + shm_size) / 1024:.1f} KB")
    print(f"  单条记忆磁盘开销:         {(db_size + wal_size + shm_size) / TOTAL:.0f} bytes")

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n{'='*64}")
    print(f"  性能基准汇总")
    print(f"{'='*64}")
    print(f"  写入 {TOTAL} 条:            {format_ms(t_total_write)} ({TOTAL / t_total_write:.0f} ops/s)")
    print(f"  单条写入 P50:               {format_ms(wstat['p50'])}")
    print(f"  Recall (有query+场域) P50:  {format_ms(rstat['p50'])}")
    print(f"  检索引擎:                   FTS5 trigram + Graph (Dual Mirror)")
    print(f"  DB 磁盘占用:                {(db_size + wal_size + shm_size) / 1024:.1f} KB")

    avg_len = mh.conn.execute("SELECT AVG(LENGTH(content)) FROM memories").fetchone()[0]
    print(f"\n平均文本长度: {avg_len:.0f} 字符")
    print(f"如果 < 30 字符，说明 280 bytes/条 是因为文本极短，正常")

    mh.close()
    print(f"\n[OK] 基准测试完成 (零 token 消耗)")
    print(f"     测试数据库已保留: {DB}")


if __name__ == "__main__":
    main()
