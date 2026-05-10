#!/usr/bin/env python3
"""
DyadCore Layer 2 — 对话记忆质量评估。

评估核心问题：有 DyadCore 记忆 vs 没有，LLM 回答质量是否有显著提升？

Usage:
    python eval_conversation.py                  # 全量评估（需 LLM）
    python eval_conversation.py --mock           # Mock 模式（无 LLM，验证框架）
    python eval_conversation.py --scenario 1     # 只跑场景 1
    python eval_conversation.py --model qwen2.5  # 指定模型

环境变量:
    DYADCORE_LLM_BASE_URL  — API 地址（默认 http://localhost:11434/v1）
    DYADCORE_LLM_API_KEY   — API Key（默认 ollama）
    DYADCORE_LLM_MODEL     — 模型名（默认 qwen2.5:7b）
"""

import argparse
import json
import math
import os
import re
import sys
import time
import tempfile
from collections import defaultdict
from typing import Optional

import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# 导入 DyadCore 和 hermes_bridge
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dyadcore import DyadCore
from hermes_bridge import format_for_prompt, should_recall, infer_field


# ===================================================================
# 常量
# ===================================================================

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """你是一个乐于助人的 AI 助手。你会看到对话历史以及用户之前分享过的背景信息。

重要规则：
1. 始终直接回答用户的当前问题，给出具体的建议或答案
2. 利用背景信息来个性化你的回答（如根据偏好做推荐）
3. 回答要具体、有针对性，包含具体细节（名字、数字、具体建议）
4. 绝对不要说"我记住了"、"了解了"、"收到"之类的确认语——直接回答即可
5. 不要复述背景信息，直接用它们来做判断"""

JUDGE_SYSTEM = """你是一个严谨的对话质量评估专家。你的任务是对比两个 AI 助手对同一问题的回答。

请仅输出 JSON，格式如下（不要加任何其他文字）：
```json
{
  "accuracy": {"A": <1-5>, "B": <1-5>, "reasoning": "<一句话>"},
  "completeness": {"A": <1-5>, "B": <1-5>, "reasoning": "<一句话>"},
  "specificity": {"A": <1-5>, "B": <1-5>, "reasoning": "<一句话>"},
  "hallucination": {"A": <true/false>, "B": <true/false>, "reasoning": "<一句话>"},
  "winner": "<A/B/tie>",
  "overall_reasoning": "<总体评价，2-3句话>"
}
```

评分标准:
- accuracy (1-5): 回答中的事实是否与 Ground Truth 一致。5=完全正确，1=严重错误
- completeness (1-5): 是否覆盖了所有相关的历史信息。5=完全覆盖，1=遗漏关键信息
- specificity (1-5): 回答是否具体。5=非常具体（人名、数字、细节），1=极其模糊
- hallucination: 是否编造了与 Ground Truth 矛盾或不存在的信息
- winner: 综合评估更好的助手（A/B/tie）"""


# ===================================================================
# 场景定义
# ===================================================================

class Scenario:
    """一个多轮对话场景。"""
    def __init__(self, sid: str, name: str, field: str, turns: list[dict],
                 memories: list[str] | None = None,
                 agent_self_memories: list[str] | None = None):
        self.sid = sid
        self.name = name
        self.field = field         # 默认 field
        self.turns = turns         # list of {role, content, is_test, ground_truth}
        self.memories = memories or []  # 预加载的话语链记忆
        self.agent_self_memories = agent_self_memories or []  # 预加载的自省链记忆（锚定）

    @property
    def test_points(self) -> list[dict]:
        return [t for t in self.turns if t.get("is_test")]

    @property
    def total_turns(self) -> int:
        return len([t for t in self.turns if t["role"] == "user"])


# ---- 各场景共用的稀释对话（模拟长对话中的无关话题，将记忆信息推向远端） ----

FILLER_TURNS = [
    {"role": "user", "content": "最近天气真是忽冷忽热的，都不知道穿什么好了。"},
    {"role": "assistant", "content": "确实，换季时节建议洋葱穿搭法，方便随时增减衣物。"},
    {"role": "user", "content": "对了，你有什么好用的笔记软件推荐吗？我平时工作需要记很多东西。"},
    {"role": "assistant", "content": "Notion 和 Obsidian 都不错，前者灵活适合团队协作，后者适合本地化知识管理。"},
    {"role": "user", "content": "中午吃什么好呢？公司附近选择挺多的但每次都纠结。"},
    {"role": "assistant", "content": "可以试试附近新开的轻食店，或者和同事一起点外卖换换口味。"},
    {"role": "user", "content": "周末想去看电影放松一下，最近有什么好片子推荐？"},
    {"role": "assistant", "content": "最近有几部不错的科幻片和悬疑片，看你喜欢什么类型。动作片也有新上的。"},
]


def _insert_fillers(turns: list[dict]) -> list[dict]:
    """在第一个测试点之前插入稀释对话，将记忆信息推远。"""
    out = []
    fillers_inserted = False
    for turn in turns:
        if turn.get("is_test") and not fillers_inserted:
            out.extend(FILLER_TURNS)
            fillers_inserted = True
        out.append(turn)
    return out

SCENARIO_1 = Scenario(
    sid="1", name="个人偏好", field="personal_info",
    memories=[
        "用户不喜欢吃甜的，特别是奶油蛋糕。但是很喜欢咖啡，每天早上必喝一杯手冲。口味偏辣，无辣不欢。",
        "用户喜欢非洲豆（耶加雪菲），果酸味。不喝意式浓缩，觉得太苦。",
        "用户工作习惯：早上 7 点到 9 点是最高效时间段，这段时间一般不接电话和开会。",
    ],
    agent_self_memories=[
        "用户饮食画像：厌甜喜辣，咖啡重度用户，偏好非洲豆手冲（耶加雪菲），拒绝意式浓缩和甜腻食品。",
        "用户工作画像：晨间7-9点为深度工作黄金时段，此期间拒绝会议和电话。",
    ],
    turns=[
        # === 测试点 ===
        {"role": "user", "content": "朋友过生日，让我帮忙挑个蛋糕。之前说过我的口味偏好，你觉得什么口味合适？",
         "is_test": True,
         "ground_truth": "用户不喜欢甜食和奶油蛋糕，应建议低糖或无奶油的选择"},

        {"role": "user", "content": "下午 3 点想喝点东西提神，还记得我之前的咖啡偏好吗？给个推荐。",
         "is_test": True,
         "ground_truth": "用户喜欢非洲豆手冲咖啡（耶加雪菲），不喜欢意式浓缩"},

        {"role": "user", "content": "团队想约我明天早上 8 点开会讨论需求，你之前知道我的工作习惯，觉得合适吗？",
         "is_test": True,
         "ground_truth": "用户早上 7-9 点是高效工作时间，不接电话不开会，应建议改时间"},

        {"role": "user", "content": "周末约了人去吃川菜，上次说过我的口味，给我推荐几道经典必点菜吧。",
         "is_test": True,
         "ground_truth": "用户口味偏辣，无辣不欢，推荐辣味川菜"}])


# ---- Scenario 2: 项目技术栈 ----

SCENARIO_2 = Scenario(
    sid="2", name="项目技术栈", field="project_tech",
    memories=[
        "数据库：PostgreSQL 15，选型理由：JSONB 支持和全文检索能力。后端：Python FastAPI。前端：Vue 3 + TypeScript（从 Vue 2 迁移中）。",
        "CI/CD：GitHub Actions，部署到 AWS ECS Fargate。API 网关：Kong。认证：OAuth2 + JWT。缓存：Redis Cluster（3主3从），用于商品详情和用户 session。",
        "日志：Elasticsearch + Fluentd。监控：Prometheus + Grafana。",
    ],
    agent_self_memories=[
        "电商后台项目全栈画像：DB=PostgreSQL15(JSONB+全文检索)，后端=Python FastAPI，前端=Vue3+TypeScript。",
        "电商后台项目基础设施：CI/CD=GitHubActions→AWS ECS Fargate，网关=Kong(OAuth2+JWT)，缓存=RedisCluster3主3从，日志=ES+Fluentd，监控=Prometheus+Grafana。",
    ],
    turns=[
        # === 测试点 ===
        {"role": "user", "content": "数据库选了什么？我想在另一个项目里参考一下。之前聊过技术栈的。",
         "is_test": True,
         "ground_truth": "PostgreSQL 15，选择理由：JSONB 支持、全文检索"},

        {"role": "user", "content": "我们 API 网关用的什么？认证方案是啥来着？上次讨论过的，我写文档需要。",
         "is_test": True,
         "ground_truth": "Kong 网关，OAuth2 + JWT 认证"},

        {"role": "user", "content": "用户说登录状态经常丢，可能是哪里的问题？记得之前说过我们的缓存架构。",
         "is_test": True,
         "ground_truth": "Redis Cluster 3主3从负责 session 存储，可能是 session 过期或 Redis 问题"},

        {"role": "user", "content": "部署流程是什么？新同事问我要怎么上线一个新功能。之前说过的。",
         "is_test": True,
         "ground_truth": "GitHub Actions CI/CD 部署到 AWS ECS Fargate"},

        {"role": "user", "content": "我在写系统架构文档，把日志和监控那块也给我总结一下。之前提过的。",
         "is_test": True,
         "ground_truth": "日志：Elasticsearch+Fluentd；监控：Prometheus+Grafana"}])


# ---- Scenario 3: 会议跟进 ----

SCENARIO_3 = Scenario(
    sid="3", name="会议跟进", field="meeting_notes",
    memories=[
        "Sprint Planning 决策1：用户权限系统重构，P0 优先级，CTO 要求下周五前完成。",
        "Sprint Planning 决策2：UI 规范变更——按钮圆角 4px→8px，蓝色统一 #3B82F6。",
        "Sprint Planning 决策3：搜索功能确定用 Elasticsearch（替代 LIKE 查询）。",
        "Sprint Planning 决策4：下版支持多语言，先做中英文。数据库迁移脚本 D-042，由用户负责。API 文档用 OpenAPI 3.0 规范。",
    ],
    agent_self_memories=[
        "Sprint规划摘要：P0=用户权限系统重构(下周五deadline)，UI规范=圆角8px+蓝#3B82F6，搜索=Elasticsearch，多语言=中英文先行，API文档=OpenAPI3.0，DB迁移=D-042(用户负责)。",
    ],
    turns=[
        # === 测试点 ===
        {"role": "user", "content": "CTO 强调的那个高优需求是什么？我忘了截止日期了。上次 sprint planning 说的。",
         "is_test": True,
         "ground_truth": "用户权限系统重构，P0 优先级，下周五截止"},

        {"role": "user", "content": "UI 那边的新规范，之前开会说的按钮和颜色具体怎么改？",
         "is_test": True,
         "ground_truth": "按钮圆角 4px→8px，蓝色统一 #3B82F6"},

        {"role": "user", "content": "搜索功能确定方案了吗？上次讨论过的，别又推翻重来。",
         "is_test": True,
         "ground_truth": "已确定用 Elasticsearch，替代 LIKE 查询"},

        {"role": "user", "content": "API 文档整理要按什么标准来做？之前会上提过的。",
         "is_test": True,
         "ground_truth": "OpenAPI 3.0 规范"},

        {"role": "user", "content": "多语言的排期是怎样的？先做哪几个语言？之前说过的。",
         "is_test": True,
         "ground_truth": "先做中英文"},

        {"role": "user", "content": "数据库迁移那个任务是谁负责的来着？脚本编号还记得吗？",
         "is_test": True,
         "ground_truth": "用户自己负责，脚本编号 D-042"}])


# ---- Scenario 4: Bug 追踪 ----

SCENARIO_4 = Scenario(
    sid="4", name="Bug 追踪", field="bugs_issues",
    memories=[
        "Bug #1：登录页密码输入框，暗黑模式下边框颜色太浅 #E5E7EB，对比度不够。改成 #6B7280。P1。",
        "Bug #2：商品详情页价格格式化，超 10000 元缺千分位分隔符，显示 12000 而非 12,000。P2。toLocaleString 可解决。",
        "Bug #3：数据库迁移脚本 D-042 预发布环境锁表超时，原因：漏设 lock_timeout。已修复。P1。",
        "事故：昨天下午 3 点 Redis 集群 master 宕机，slave 切换花 45 秒，期间用户 session 全丢。",
    ],
    agent_self_memories=[
        "Bug列表：P1=暗黑模式密码框边框#E5E7EB→#6B7280(待修)；P1=D-042锁表超时漏设lock_timeout(已修复)；P2=价格千分位格式化toLocaleString。",
        "事故记录：Redis master宕机→slave切换45s→session全丢。",
    ],
    turns=[
        # === 测试点 ===
        {"role": "user", "content": "暗黑模式那个 UI bug 是什么来着？之前整理过的，解决了吗？",
         "is_test": True,
         "ground_truth": "登录密码框边框颜色太浅 #E5E7EB→#6B7280，P1"},

        {"role": "user", "content": "价格显示的 bug 优先级多少？之前提过的，改动复杂吗？",
         "is_test": True,
         "ground_truth": "P2，千分位格式化问题，toLocaleString 可解决"},

        {"role": "user", "content": "D-042 那个脚本出过什么问题？上次说过的，现在修好了吗？",
         "is_test": True,
         "ground_truth": "锁表超时，漏设 lock_timeout，已修复"},

        {"role": "user", "content": "上次生产环境那个事故是什么原因？影响范围多大？",
         "is_test": True,
         "ground_truth": "Redis master 宕机，切换 45 秒，用户 session 全丢"},

        {"role": "user", "content": "现在线上还有哪些 P1 的 bug 没修？帮我列一下。",
         "is_test": True,
         "ground_truth": "Bug #1 暗黑模式密码框 (P1) 和 Bug #3 D-042 迁移问题 (P1，已修复)"}])


# ---- Scenario 5: 跨场综合 ----

SCENARIO_5 = Scenario(
    sid="5", name="跨场综合", field="personal_info",
    memories=[
        "技术栈：后端 Python FastAPI + PostgreSQL 15，前端 Vue 3 + TypeScript。在学 Rust。",
        "CI/CD：GitHub Actions，部署到 AWS ECS Fargate。API 网关：Kong，认证：OAuth2 + JWT。",
        "用户是远程工作者，偏好异步沟通，不希望突然的语音/视频电话，希望提前一天预约。早上 7-9 点完全不接会议。",
        "用户偏后端开发，前端 Vue 3 + TS 也能写但不熟。",
    ],
    agent_self_memories=[
        "用户技术画像：后端Python FastAPI+PostgreSQL15为主，在学Rust。前端Vue3+TS能写不熟。基础设施：GitHubActions→AWS ECS，Kong+OAuth2+JWT。",
        "用户协作画像：远程工作者，强烈偏好异步沟通（文字优先），拒绝突发的语音/视频，要求提前一天预约。晨间7-9点为不可侵犯的深度工作时段。",
    ],
    turns=[
        # === 测试点 ===
        {"role": "user", "content": "基于我的技术栈（后端 Python+PostgreSQL，在学 Rust），之前聊过的，你觉得下一个值得深入的技术方向是什么？",
         "is_test": True,
         "ground_truth": "结合已有的 Python+PostgreSQL 和在学的 Rust，推荐方向应考虑用户实际技术栈"},

        {"role": "user", "content": "有同事说下午想跟我快速语音沟通一个紧急需求，你知道我的沟通偏好，我该怎么回复？",
         "is_test": True,
         "ground_truth": "用户偏好异步沟通，不希望突然的电话，建议文字沟通或提前预约"},

        {"role": "user", "content": "我有个想法：能不能在后端用 Rust 重写一些对性能要求高的 API 端点？记得之前说过的技术栈，你觉得怎么样？",
         "is_test": True,
         "ground_truth": "用户正在学 Rust，项目后端 Python FastAPI，部署在 AWS ECS，可以考虑 Rust 微服务"},

        {"role": "user", "content": "团队说想搞个每天的晨会，固定在早上 8 点。你知道我的习惯，觉得这个时间好吗？",
         "is_test": True,
         "ground_truth": "用户早上 7-9 点不接会议，8 点不合适"},

        {"role": "user", "content": "有人建议我把 API 网关从 Kong 换成 Traefik。之前聊过我们的架构，你觉得值得吗？",
         "is_test": True,
         "ground_truth": "现有网关 Kong + OAuth2 JWT 已稳定运行，迁移成本需要考虑"}])


# ---- Scenario 6: 偏好修正 ----
# 测试质量链能否区分"已过时"和"当前有效"的偏好信息

SCENARIO_6 = Scenario(
    sid="6", name="偏好修正", field="personal_info",
    memories=[
        "用户过去重度依赖咖啡，每天两三杯意式浓缩，喜欢深度烘焙的苦味。",
        "最近体检后用户决定大幅减少咖啡因摄入，改喝低因咖啡和花草茶了。",
        "用户以前无辣不欢，每周必吃川菜火锅，最爱上火的重辣牛油锅底。",
        "最近胃不舒服，医生建议清淡饮食，用户开始偏好粤菜和蒸菜，少油少盐。",
        "用户原来用 VS Code 作为主力编辑器，装了很多插件。",
        "但最近半年逐渐切换到 Neovim，现在已经完全不用 VS Code 了。",
    ],
    agent_self_memories=[
        "用户健康驱动的偏好更新(2025年)：咖啡意式浓缩→低因/花草茶，饮食川辣火锅→粤蒸清淡，编辑器VSCode→Neovim。更新原因：体检+胃病，属近期持续变化。",
    ],
    turns=[
        {"role": "user", "content": "下午犯困想喝杯咖啡，还记得我最近的健康状况和咖啡偏好吗？推荐个合适的。",
         "is_test": True,
         "ground_truth": "用户已因健康原因改喝低因咖啡和花草茶，不应推荐意式浓缩或高咖啡因饮品"},

        {"role": "user", "content": "同事约我周末去吃重庆火锅，你知道我最近的饮食状况，觉得该去吗？",
         "is_test": True,
         "ground_truth": "用户最近胃不好已改清淡饮食（粤菜蒸菜），麻辣火锅不适合"},

        {"role": "user", "content": "新同事问我用什么编辑器，我换了有一阵子了。之前跟你聊过的。",
         "is_test": True,
         "ground_truth": "用户已从 VS Code 切换到 Neovim，不再使用 VS Code"},

        {"role": "user", "content": "对比一下我以前和现在的饮食习惯，有什么变化？为什么会变？",
         "is_test": True,
         "ground_truth": "从川辣重口变为粤蒸清淡，原因是胃病医生建议"},

        {"role": "user", "content": "我想点个健康的外卖午餐，记得我最近的饮食偏好吗？给个建议。",
         "is_test": True,
         "ground_truth": "推荐粤菜蒸菜类，少油少盐，避免川辣"}])

# ---- Scenario 7: 技术栈迁移 ----
# 测试质量链能否追踪技术决策的演进历史（而非仅记住最新状态）

SCENARIO_7 = Scenario(
    sid="7", name="技术栈迁移", field="project_tech",
    memories=[
        "项目早期（2023 Q1）：后端 Express.js + MongoDB，部署在 Heroku，前端 jQuery + Bootstrap。",
        "2024 Q2 迁移：后端重写为 Python FastAPI + PostgreSQL 15，部署迁移到 AWS ECS Fargate。",
        "迁移原因：MongoDB 不适合复杂事务场景，Express.js 缺乏类型安全，Heroku 成本过高。",
        "API 网关演进：最初 Nginx + 手动配置 → 2024年改用 Kong + OAuth2 + JWT。",
        "前端演进：jQuery(2023) → Vue 2(2024 Q1) → Vue 3 + TypeScript(2025 Q1)。每次迁移渐进完成。",
        "监控体系也是逐步建立的：早期无监控 → 2024 Q2 Prometheus + Grafana → 2025 Q1 加入 OpenTelemetry 分布式追踪。",
    ],
    agent_self_memories=[
        "项目演进历程(2023-2025)：后端Express.js/MongoDB/Heroku→FastAPI/PostgreSQL/AWS ECS。前端jQuery→Vue2→Vue3+TS。网关Nginx→Kong+OAuth2 JWT。监控无→Prometheus+Grafana→+OpenTelemetry。核心迁移驱动：类型安全、事务ACID、成本优化。",
    ],
    turns=[
        {"role": "user", "content": "现在数据库用的是什么？我记得从 MongoDB 迁走过。之前聊过整个迁移历史的。",
         "is_test": True,
         "ground_truth": "当前 PostgreSQL 15，2024年从 MongoDB 迁移，原因是不适合复杂事务"},

        {"role": "user", "content": "新来的架构师问：为什么当初从 Heroku 迁到 AWS？你帮我总结一下原因。",
         "is_test": True,
         "ground_truth": "Heroku 成本过高，AWS ECS Fargate 更灵活且成本更低"},

        {"role": "user", "content": "前端技术栈经历了哪些演变？我之前跟你说过完整的路线。",
         "is_test": True,
         "ground_truth": "jQuery → Vue 2 → Vue 3 + TypeScript，渐进式迁移"},

        {"role": "user", "content": "帮我理一下 API 网关的演进路线和现在的认证方案。上次说过的。",
         "is_test": True,
         "ground_truth": "Nginx 手动配置 → Kong + OAuth2 + JWT"},

        {"role": "user", "content": "监控和可观测性这块，我们经历了哪些阶段？现在是什么方案？",
         "is_test": True,
         "ground_truth": "无监控 → Prometheus+Grafana → 加入 OpenTelemetry 分布式追踪"}])

# ---- Scenario 8: 稀疏信息聚合 ----
# 测试质量链能否将散落在不同话题中的碎片信息聚合成完整画像

SCENARIO_8 = Scenario(
    sid="8", name="稀疏信息聚合", field="general",
    memories=[
        "用户开发的项目叫 DyadCore，是一个本地优先的个人记忆系统，开源在 GitHub 上，MIT 协议。",
        "用户偏好的开发环境：Windows 11 + WSL2（Ubuntu），Python 3.12，用 Neovim 做主力编辑器。",
        "DyadCore 技术核心：SQLite 单文件存储 + FTS5 trigram 全文搜索 + sqlite-vec 向量检索，零外部服务依赖。",
        "用户写了详细的单元测试，覆盖率目标是 85%+，但觉得集成测试太重不愿多写。",
        "DyadCore 的双螺旋架构：事实链（utterance 话语记录）+ 质量链（anchored agent_self 反思蒸馏），Peer 对等共建。",
        "用户的沟通风格：喜欢直接、简洁的反馈，不要客套话。Code review 时关注逻辑正确性多于风格问题。",
        "目前 DyadCore eval 已完成 Layer 1（检索质量，72.58% Hit Rate）和 Layer 2（对话质量，80%+ Win Rate），正在优化质量链的锚定相关性。",
    ],
    agent_self_memories=[
        "DyadCore项目全景：本地优先记忆系统，MIT开源，SQLite+FTS5+sqlite-vec，双螺旋架构(事实链+质量链)。开发环境Win11+WSL2/Py3.12/Neovim。测试倾向单测85%覆盖，轻集成测试。沟通偏好直接简洁。Eval进展Layer1(72.58%)+Layer2(80%+WinRate)已完成。",
    ],
    turns=[
        {"role": "user", "content": "给我的开源项目写个 pitch，一两句话概括它是什么、用什么技术。之前聊过很多细节的。",
         "is_test": True,
         "ground_truth": "DyadCore 是本地优先的个人记忆系统，SQLite+FTS5+sqlite-vec，双螺旋架构，零外部依赖，MIT 开源"},

        {"role": "user", "content": "有贡献者问项目的测试策略是什么。我之前说过我的偏好，你帮我回复一下。",
         "is_test": True,
         "ground_truth": "重视单元测试覆盖率 85%+，集成测试尽量少写"},

        {"role": "user", "content": "我想在项目 README 里加个 badge 展示 eval 进展。我们的 eval 做到哪了？之前讨论过的。",
         "is_test": True,
         "ground_truth": "Layer 1 检索质量 72.58% + Layer 2 对话质量 80%+ Win Rate 已完成"},

        {"role": "user", "content": "新加入的协作者说我的 code review 评论太直接了。你知道我的沟通风格，帮我解释一下。",
         "is_test": True,
         "ground_truth": "用户偏好直接简洁的反馈，关注逻辑正确性多于代码风格"},

        {"role": "user", "content": "帮我总结一下 DyadCore 的架构设计理念，特别是双螺旋部分。我跟你说过很多次了。",
         "is_test": True,
         "ground_truth": "双螺旋架构：事实链（utterance 话语记录）+ 质量链（agent_self 反思蒸馏），Peer 对等共建"}])

ALL_SCENARIOS = [SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5, SCENARIO_6, SCENARIO_7, SCENARIO_8]


# ===================================================================
# LLM Client
# ===================================================================

class LLMClient:
    """OpenAI 兼容 API 客户端。"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 api_key: str = DEFAULT_API_KEY,
                 model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _call_api(self, messages: list[dict], temperature: float = 0.7,
                  max_tokens: int = 1024) -> str:
        """调用 Chat Completions API，含重试逻辑。"""
        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        last_error = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=body, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                })
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM API error {e.code}: {err_body[:500]}")
                if e.code == 429:
                    time.sleep(2 ** attempt)
                else:
                    raise last_error
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"LLM API connection error: {e.reason}")
                if attempt < 2:
                    time.sleep(1)
        raise last_error

    def chat(self, history: list[dict], memory_context: str = "",
             system_prompt: str = SYSTEM_PROMPT, temperature: float = 0.7) -> str:
        """生成助手回复。

        Args:
            history: [{"role": "user"/"assistant", "content": "..."}]
                     最后一条必须是当前用户问题！
            memory_context: format_for_prompt() 的输出（可为空）
            system_prompt: 系统提示词
            temperature: 采样温度
        """
        messages = [{"role": "system", "content": system_prompt}]
        if memory_context:
            messages.append({
                "role": "system",
                "content": f"用户之前分享过的背景信息：\n\n{memory_context}"
            })
        messages.extend(history)
        return self._call_api(messages, temperature=temperature)

    def judge(self, question: str, response_a: str, response_b: str,
              ground_truth: str, scenario_summary: str = "") -> dict:
        """LLM-as-Judge 对比评估两个回答。"""
        prompt = f"""## 对话背景
{scenario_summary or "无额外背景"}

## 应被记忆的关键信息（Ground Truth）
{ground_truth}

## 用户问题
{question}

## 助手 A 的回答
{response_a}

## 助手 B 的回答
{response_b}

请对比评估两个回答。仅输出 JSON。"""

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        raw = self._call_api(messages, temperature=0.1, max_tokens=1024)
        return self._parse_judge_output(raw)

    def _parse_judge_output(self, raw: str) -> dict:
        """从 LLM 输出中提取 JSON。"""
        # 尝试匹配 ```json ... ``` 或直接 JSON
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        json_str = m.group(1).strip() if m else raw.strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 也许 LLM 输出有额外的文字，尝试找到第一个 { 和最后一个 }
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(json_str[start:end])
            return {"error": "JSON parse failed", "raw": raw}


# ===================================================================
# Mock LLM（无 LLM 时验证框架用）
# ===================================================================

class MockLLMClient(LLMClient):
    """模拟 LLM，基于 ground truth 规则生成回复。"""

    def __init__(self):
        super().__init__(base_url="mock://", api_key="mock", model="mock")

    def _call_api(self, messages, temperature=0.7, max_tokens=1024):
        return "mock_response"

    def chat(self, history, memory_context="", system_prompt="", temperature=0.7):
        last_msg = history[-1]["content"] if history else ""
        if memory_context:
            return f"[WITH MEMORY] 基于记忆回答了: {last_msg[:40]}..."
        else:
            return f"[WITHOUT MEMORY] 泛泛回答了: {last_msg[:40]}..."

    def judge(self, question, response_a, response_b, ground_truth, scenario_summary=""):
        # Mock judge: A (with memory) 总是更好
        a_has_memory = "[WITH MEMORY]" in response_a
        b_has_memory = "[WITH MEMORY]" in response_b
        return {
            "accuracy": {
                "A": 4 if a_has_memory else 2,
                "B": 4 if b_has_memory else 2,
                "reasoning": "A 包含记忆信息" if a_has_memory else "B 包含记忆信息"
            },
            "completeness": {
                "A": 4 if a_has_memory else 1,
                "B": 4 if b_has_memory else 1,
                "reasoning": ""
            },
            "specificity": {
                "A": 5 if a_has_memory else 2,
                "B": 5 if b_has_memory else 2,
                "reasoning": ""
            },
            "hallucination": {
                "A": not a_has_memory,
                "B": not b_has_memory,
                "reasoning": ""
            },
            "winner": "A" if a_has_memory else ("B" if b_has_memory else "tie"),
            "overall_reasoning": "Mock judge: 有记忆的助手提供更准确的回答"
        }


# ===================================================================
# 场景执行引擎
# ===================================================================

class ScenarioRunner:
    """运行单个场景的双路对比：Path A (有记忆) / Path B (无记忆)。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, scenario: Scenario) -> dict:
        """运行场景，返回测试点配对结果。"""
        responses_a, recall_stats = self._run_path_a(scenario)
        responses_b = self._run_path_b(scenario)
        test_points = self._pair(responses_a, responses_b)

        return {
            "scenario_id": scenario.sid,
            "scenario_name": scenario.name,
            "field": scenario.field,
            "total_turns": scenario.total_turns,
            "test_points": test_points,
            "recall_triggers": recall_stats["triggers"],
            "recall_hits": recall_stats["hits"],
            "recall_trigger_rate": (
                recall_stats["triggers"] / scenario.total_turns
                if scenario.total_turns > 0 else 0
            ),
        }

    def _pair(self, responses_a: list[dict], responses_b: list[dict]) -> list[dict]:
        """按位置配对两路响应。"""
        test_points = []
        for i, ra in enumerate(responses_a):
            rb = responses_b[i] if i < len(responses_b) else {}
            test_points.append({
                "question": ra["question"],
                "ground_truth": ra["ground_truth"],
                "response_a": ra["response"],
                "response_b": rb.get("response", ""),
            })
        return test_points

    def _run_path_a(self, scenario: Scenario) -> tuple[list[dict], dict]:
        """Path A: With Memory — 预加载记忆 + 实时召回。

        Dual Mirror：用户和 Agent 的记忆作为对等痕迹写入同一 field。
        reflections 表自动发现记忆间的语义关系。
        """
        mh = self._create_dyadcore()

        # 写入用户痕迹
        for mem in scenario.memories:
            mh.write(mem, memory_type="utterance", source="user",
                     field=scenario.field)

        # 写入 Agent 的观察记录（对等的 agent 痕迹，非特殊类别）
        for mem in scenario.agent_self_memories:
            mh.write(mem, memory_type="utterance", source="agent",
                     field=scenario.field)

        history = []
        field = scenario.field
        test_responses = []
        triggers = 0
        hits = 0

        for turn in scenario.turns:
            if turn["role"] == "user":
                # 召回
                should_rec = should_recall(history)
                memory_context = ""
                if should_rec:
                    triggers += 1
                    results = mh.recall(
                        query=turn["content"],
                        field_hint=field,
                        limit=5
                    )
                    if results:
                        memory_context = format_for_prompt(results, dyadcore=mh)
                        gt = turn.get("ground_truth", "")
                        if gt and any(self._crude_match(gt, r["content"])
                                      for r in results):
                            hits += 1

                # 生成回复（先添加当前用户消息到 history）
                history.append({"role": "user", "content": turn["content"]})
                response = self.llm.chat(history, memory_context)
                history.append({"role": "assistant", "content": response})

                # 跨场 field 推断
                if scenario.sid == "5":
                    inferred = infer_field(history, field)
                    if inferred and inferred != field:
                        field = inferred

                # 收集测试点
                if turn.get("is_test"):
                    test_responses.append({
                        "question": turn["content"],
                        "ground_truth": turn.get("ground_truth", ""),
                        "response": response,
                    })
            elif turn["role"] == "assistant":
                history.append({"role": "assistant", "content": turn["content"]})

        mh.close()
        return test_responses, {"triggers": triggers, "hits": hits}

    def _run_path_b(self, scenario: Scenario,
                    max_history_messages: int = 4) -> list[dict]:
        """Path B: Without Memory — 无记忆基线（截断历史，模拟长对话中上下文被稀释）。

        Args:
            max_history_messages: Path B 最多保留的历史消息数（模拟有限上下文窗口）。
        """
        history = []
        test_responses = []

        for turn in scenario.turns:
            if turn["role"] == "user":
                # 只使用最近的历史（不含记忆检索），先添加当前用户消息
                history.append({"role": "user", "content": turn["content"]})
                recent = history[-max_history_messages:] if max_history_messages > 0 else history
                response = self.llm.chat(recent, "")
                history.append({"role": "assistant", "content": response})

                if turn.get("is_test"):
                    test_responses.append({
                        "question": turn["content"],
                        "ground_truth": turn.get("ground_truth", ""),
                        "response": response,
                    })
            elif turn["role"] == "assistant":
                history.append({"role": "assistant", "content": turn["content"]})

        return test_responses

    def _create_dyadcore(self) -> DyadCore:
        """创建临时 DyadCore 实例。"""
        # 使用命名临时文件（ensure file stays valid for the duration）
        fd, path = tempfile.mkstemp(suffix=".db", prefix="dyadcore_eval_")
        os.close(fd)
        return DyadCore(path)

    @staticmethod
    def _crude_match(ground_truth: str, memory_content: str) -> bool:
        """简陋的匹配：检查 ground truth 关键词是否在记忆中。"""
        # 提取 ground_truth 中的关键词（中文 2-4 字词）
        keywords = []
        for ch in ground_truth:
            if '\u4e00' <= ch <= '\u9fff':
                keywords.append(ch)
        keyword_str = ''.join(keywords[:8])  # 前 8 个汉字
        if len(keyword_str) < 2:
            return False
        # 2-gram 匹配
        hits = 0
        for i in range(len(keyword_str) - 1):
            if keyword_str[i:i+2] in memory_content:
                hits += 1
        return hits >= len(keyword_str) // 3


# ===================================================================
# Judge 评估器
# ===================================================================

class JudgeEvaluator:
    """批量调用 LLM Judge，汇总指标。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, scenario_results: list[dict]) -> dict:
        """评估所有场景的测试点。"""
        all_judgments = []
        for sr in scenario_results:
            for tp in sr["test_points"]:
                if not tp.get("response_a") or not tp.get("response_b"):
                    continue
                try:
                    judgment = self.llm.judge(
                        question=tp["question"],
                        response_a=tp["response_a"],
                        response_b=tp["response_b"],
                        ground_truth=tp["ground_truth"],
                        scenario_summary=sr["scenario_name"]
                    )
                    judgment["scenario_id"] = sr["scenario_id"]
                    judgment["scenario_name"] = sr["scenario_name"]
                    judgment["question"] = tp["question"]
                    judgment["ground_truth"] = tp["ground_truth"]
                    all_judgments.append(judgment)
                except Exception as e:
                    all_judgments.append({
                        "scenario_id": sr["scenario_id"],
                        "scenario_name": sr["scenario_name"],
                        "question": tp["question"],
                        "error": str(e),
                    })

        metrics = self._compute_metrics(all_judgments)
        metrics["judgments"] = all_judgments
        metrics["total_judgments"] = len(all_judgments)
        return metrics

    def _compute_metrics(self, judgments: list[dict]) -> dict:
        """从所有 Judge 结果中计算汇总指标。"""
        scores = {"A": {"accuracy": [], "completeness": [], "specificity": []},
                  "B": {"accuracy": [], "completeness": [], "specificity": []}}
        hallucination = {"A": 0, "B": 0, "total": 0}
        winners = {"A": 0, "B": 0, "tie": 0}
        per_scenario = defaultdict(lambda: {
            "scores_a": {"accuracy": [], "completeness": [], "specificity": []},
            "scores_b": {"accuracy": [], "completeness": [], "specificity": []},
            "winners": {"A": 0, "B": 0, "tie": 0},
        })

        for j in judgments:
            if "error" in j:
                continue

            sid = j.get("scenario_id", "?")
            for dim in ["accuracy", "completeness", "specificity"]:
                a_val = j.get(dim, {}).get("A", 0)
                b_val = j.get(dim, {}).get("B", 0)
                scores["A"][dim].append(a_val)
                scores["B"][dim].append(b_val)
                per_scenario[sid]["scores_a"][dim].append(a_val)
                per_scenario[sid]["scores_b"][dim].append(b_val)

            hal_a = j.get("hallucination", {}).get("A", False)
            hal_b = j.get("hallucination", {}).get("B", False)
            if hal_a:
                hallucination["A"] += 1
            if hal_b:
                hallucination["B"] += 1
            hallucination["total"] += 1

            winner = j.get("winner", "tie")
            if winner in winners:
                winners[winner] += 1
                if sid in per_scenario:
                    per_scenario[sid]["winners"][winner] += 1

        n = len([j for j in judgments if "error" not in j]) or 1

        def mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "global": {
                "A_accuracy": mean(scores["A"]["accuracy"]),
                "B_accuracy": mean(scores["B"]["accuracy"]),
                "A_completeness": mean(scores["A"]["completeness"]),
                "B_completeness": mean(scores["B"]["completeness"]),
                "A_specificity": mean(scores["A"]["specificity"]),
                "B_specificity": mean(scores["B"]["specificity"]),
                "A_hallucination_rate": hallucination["A"] / n,
                "B_hallucination_rate": hallucination["B"] / n,
                "win_rate_a": winners["A"] / n,
                "win_rate_b": winners["B"] / n,
                "tie_rate": winners["tie"] / n,
            },
            "per_scenario": dict(per_scenario),
            "winners": winners,
            "hallucination": hallucination,
            "raw_scores": scores,
        }


# ===================================================================
# 主评估类
# ===================================================================

class ConversationEval:
    """Layer 2 评估主类。"""

    def __init__(self, llm: LLMClient, scenarios: list[Scenario] = ALL_SCENARIOS):
        self.llm = llm
        self.scenarios = scenarios
        self.runner = ScenarioRunner(llm)
        self.evaluator = JudgeEvaluator(llm)

    def run(self) -> dict:
        print_banner()
        self._print_config()

        # Phase 1: 场景执行
        print("\n" + "-" * 72)
        print("  Phase 1: 场景执行")
        print("-" * 72 + "\n")

        results_all = []

        for i, scenario in enumerate(self.scenarios, 1):
            print(f"  场景 {i}/{len(self.scenarios)}: {scenario.name} "
                  f"({scenario.total_turns} 轮, {len(scenario.test_points)} 测试点)")
            t0 = time.time()

            sr = self.runner.run(scenario)
            ts = time.time() - t0
            print(f"    召回 {sr['recall_triggers']}/{sr['total_turns']}"
                  f" ({sr['recall_trigger_rate']:.0%}),"
                  f" 命中 {sr['recall_hits']}/{sr['recall_triggers']}"
                  f" ({ts:.1f}s)")

            results_all.append(sr)

        total_test_points = sum(len(r["test_points"]) for r in results_all)
        print(f"\n  总计: {len(self.scenarios)} 场景, {total_test_points} 测试点")

        # Phase 2: Judge 评估
        print("\n" + "-" * 72)
        print("  Phase 2: Judge 评估")
        print("-" * 72 + "\n")

        metrics = self.evaluator.evaluate(results_all)

        # Phase 3: 对比分析
        print("\n" + "=" * 72)
        print("  路径对比：有记忆 (A) vs 无记忆 (B)")
        print("=" * 72 + "\n")

        self._print_comparison(metrics["global"])
        self._print_per_scenario(metrics)

        # Phase 3: 定性分析
        print("\n" + "-" * 72)
        print("  Phase 3: 定性分析")
        print("-" * 72 + "\n")

        self._print_case_studies(metrics.get("judgments", []))
        self._print_recall_analysis(results_all)

        # 总结
        print("\n" + "=" * 72)
        print("  总结")
        print("=" * 72 + "\n")

        g = metrics["global"]
        print(f"  Win Rate:           A={g['win_rate_a']:.1%}  B={g['win_rate_b']:.1%}")
        print(f"  Accuracy Δ:         +{g['A_accuracy'] - g['B_accuracy']:.2f}")
        print(f"  Completeness Δ:     +{g['A_completeness'] - g['B_completeness']:.2f}")
        print(f"  Specificity Δ:      +{g['A_specificity'] - g['B_specificity']:.2f}")
        print(f"  幻觉降低:           "
              f"{(g['B_hallucination_rate'] - g['A_hallucination_rate']) * 100:.1f}pp")
        grade = self._grade(g["win_rate_a"])
        print(f"  等级:               {grade}")
        print()

        return {"metrics": metrics}

    def _print_config(self):
        print(f"\n  LLM Model: {self.llm.model} @ {self.llm.base_url}")

    def _print_metrics(self, g: dict):
        print("  指标                    A (With Memory)  B (Without Memory)  Δ")
        print("  " + "-" * 63)
        for label, key in [("Accuracy", "accuracy"), ("Completeness", "completeness"),
                           ("Specificity", "specificity")]:
            a = g[f"A_{key}"]
            b = g[f"B_{key}"]
            d = a - b
            print(f"  {label:20s}      {a:.2f}            {b:.2f}               {'+' if d >= 0 else ''}{d:.2f}")

        ha = g["A_hallucination_rate"]
        hb = g["B_hallucination_rate"]
        print(f"  {'Hallucination Rate':20s}   {ha:.1%}            {hb:.1%}              "
              f"{'+' if hb-ha >= 0 else ''}{hb-ha:.1%}")

        print(f"\n  Win Rate:      A={g['win_rate_a']:.1%}  "
              f"B={g['win_rate_b']:.1%}  Tie={g['tie_rate']:.1%}")

    def _print_per_scenario(self, metrics: dict):
        print("\n  --- 分场景分析 ---\n")
        print(f"  {'场景':12s} {'A Acc':>6s} {'B Acc':>6s} {'Δ':>7s}  {'Win%':>6s}")
        print("  " + "-" * 45)
        for sr in self.scenarios:
            ps = metrics["per_scenario"].get(sr.sid, {})
            sa = ps.get("scores_a", {})
            sb = ps.get("scores_b", {})
            a_acc = sum(sa.get("accuracy", [0])) / max(len(sa.get("accuracy", [1])), 1)
            b_acc = sum(sb.get("accuracy", [0])) / max(len(sb.get("accuracy", [1])), 1)
            w = ps.get("winners", {"A": 0, "B": 0, "tie": 0})
            total = sum(w.values()) or 1
            print(f"  {sr.name:12s} {a_acc:6.2f} {b_acc:6.2f} "
                  f"{a_acc-b_acc:+7.2f}  {w['A']/total:6.1%}")

    def _print_case_studies(self, judgments: list[dict]):
        valid = [j for j in judgments if "error" not in j and "winner" in j]
        a_wins = [j for j in valid if j.get("winner") == "A"]
        b_wins = [j for j in valid if j.get("winner") == "B"]
        ties = [j for j in valid if j.get("winner") == "tie"]

        print("  [典型案例 — 记忆显著提升]")
        for j in a_wins[:3]:
            acc_a = j.get("accuracy", {}).get("A", "?")
            acc_b = j.get("accuracy", {}).get("B", "?")
            print(f"\n  查询: {j['question'][:80]}")
            print(f"  场景: {j['scenario_name']}")
            print(f"  Acc: A={acc_a} B={acc_b} | Winner: A")
            print(f"  GT: {j['ground_truth'][:100]}")

        print("\n  [失败案例 — 记忆反而更差]")
        if b_wins:
            for j in b_wins[:3]:
                print(f"\n  查询: {j['question'][:80]}")
                print(f"  场景: {j['scenario_name']}")
                print(f"  Winner: B | Reasoning: {j.get('overall_reasoning', 'N/A')[:100]}")
        else:
            print("  (无)")

        print(f"\n  [数据统计]")
        print(f"  A 胜: {len(a_wins)} | B 胜: {len(b_wins)} | 持平: {len(ties)} | "
              f"错误: {len(judgments) - len(valid)}")

    def _print_recall_analysis(self, scenario_results: list[dict]):
        print("\n  [召回分析]")
        for sr in scenario_results:
            print(f"  {sr['scenario_name']}: "
                  f"触发={sr['recall_triggers']}/{sr['total_turns']} "
                  f"({sr['recall_trigger_rate']:.0%}), "
                  f"命中={sr.get('recall_hits', '-')}/{sr['recall_triggers']}"
                  if sr['recall_triggers'] > 0 else
                  f"  {sr['scenario_name']}: 触发=0/{sr['total_turns']}")

    def _print_comparison(self, g: dict):
        """打印对比表：有记忆 vs 无记忆基线。"""
        print("  指标              A (With Memory)  B (Without Memory)  Δ")
        print("  " + "-" * 63)
        for label, key in [("Accuracy", "accuracy"), ("Completeness", "completeness"),
                           ("Specificity", "specificity")]:
            a = g[f"A_{key}"]
            b = g[f"B_{key}"]
            d = a - b
            print(f"  {label:20s}      {a:.2f}            {b:.2f}               {'+' if d >= 0 else ''}{d:.2f}")

        ha = g["A_hallucination_rate"]
        hb = g["B_hallucination_rate"]
        print(f"  {'Hallucination':20s}   {ha:.1%}            {hb:.1%}              "
              f"{'-' if hb-ha >= 0 else '+'}{abs(hb-ha):.1%}")

        print(f"\n  Win Rate:      A={g['win_rate_a']:.1%}  "
              f"B={g['win_rate_b']:.1%}  Tie={g['tie_rate']:.1%}")


    def _grade(self, win_rate: float) -> str:
        if win_rate >= 0.80:
            return "A (>80% win rate)"
        elif win_rate >= 0.65:
            return "B (65-80% win rate)"
        elif win_rate >= 0.50:
            return "C (50-65% win rate)"
        else:
            return "D (<50% win rate)"


# ===================================================================
# 入口
# ===================================================================

def print_banner():
    print("=" * 72)
    print("  DyadCore 对话记忆质量评估 (Layer 2)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="DyadCore Layer 2 — 对话记忆质量评估"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Mock 模式：不调用 LLM，用规则模拟回复")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="只跑指定场景 (1-8)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"指定模型（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API 地址")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key")
    args = parser.parse_args()

    # LLM 配置
    base_url = args.base_url or os.environ.get("DYADCORE_LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key = args.api_key or os.environ.get("DYADCORE_LLM_API_KEY", DEFAULT_API_KEY)
    model = args.model or os.environ.get("DYADCORE_LLM_MODEL", DEFAULT_MODEL)

    if args.mock:
        print_banner()
        print("\n  [Mock 模式] 使用模拟 LLM，验证框架结构...\n")
        llm = MockLLMClient()
    else:
        # 快速连接测试
        print_banner()
        print(f"\n  验证 LLM 连接: {model} @ {base_url} ... ", end="", flush=True)
        try:
            llm = LLMClient(base_url=base_url, api_key=api_key, model=model)
            test_resp = llm.chat(
                history=[{"role": "user", "content": "回复：OK"}],
                memory_context="",
                temperature=0.0
            )
            print("OK")
        except Exception as e:
            print(f"失败\n  {e}")
            print("\n  建议：使用 --mock 模式先跑通框架，或设置环境变量：")
            print("    DYADCORE_LLM_BASE_URL  (默认: http://localhost:11434/v1)")
            print("    DYADCORE_LLM_API_KEY   (默认: ollama)")
            print("    DYADCORE_LLM_MODEL     (默认: qwen2.5:7b)")
            sys.exit(1)

    # 选择场景
    if args.scenario:
        scenarios = [s for s in ALL_SCENARIOS if s.sid == str(args.scenario)]
    else:
        scenarios = ALL_SCENARIOS

    # 运行评估
    evaluator = ConversationEval(llm, scenarios)
    metrics = evaluator.run()

    # 保存结果
    out_path = os.path.join(_HERE, "eval_conversation_results.json")
    all_metrics = metrics.get("metrics", {})
    save_data = {
        "config": {"model": model, "base_url": base_url, "mock": args.mock},
        "global": all_metrics.get("global", {}),
        "per_scenario": {
            k: {"winners": v.get("winners", {}),
                "scores_a": {dim: sum(vals) / max(len(vals), 1)
                             for dim, vals in v.get("scores_a", {}).items()},
                "scores_b": {dim: sum(vals) / max(len(vals), 1)
                             for dim, vals in v.get("scores_b", {}).items()}}
            for k, v in all_metrics.get("per_scenario", {}).items()
        },
        "total_judgments": all_metrics.get("total_judgments", 0),
        "sample_judgments": [
            {"question": j.get("question", "")[:80],
             "winner": j.get("winner", "?"),
             "accuracy": j.get("accuracy", {}),
             "judge_reasoning": j.get("overall_reasoning", "")[:120],
             "error": j.get("error", "")}
            for j in all_metrics.get("judgments", [])[:10]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
