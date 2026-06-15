#!/usr/bin/env python3
"""
多会话偏好修正 Benchmark Runner

从 JSON 场景文件加载，评估图拓扑渲染（contradicted + echoed + field 强度）。
记忆不再随时间衰减，锚定不再参与 ranking——排序完全由查询匹配 + 图结构决定。

Usage:
    python eval_multisession.py                          # 默认 JSON
    python eval_multisession.py --dry-run                 # 仅检索排序
    python eval_multisession.py --scenarios my_test.json  # 自定义场景
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dyadcore import DyadCore
from hermes_bridge import format_for_prompt

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"
DB_PATH = "eval_multisession.db"

MCQ_SYSTEM_PROMPT = """You are an assistant answering questions about a user based on retrieved memories.

Rules:
1. Read the retrieved memories carefully — they contain the user's stated preferences, facts, and history
2. Pay attention to memory age and evolution chains — newer information may supersede older information
3. Pick the SINGLE best answer (a, b, c, or d)
4. Output ONLY the letter in parentheses, like "(a)" or "(b)"
5. Do not explain or add any other text"""

# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 api_key: str = DEFAULT_API_KEY,
                 model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def ask_mcq(self, question: str, options: list[str], memory_context: str) -> str:
        opts_text = "\n".join(options)
        prompt = f"""## Retrieved User Memories

{memory_context}

## Question

{question}

## Options

{opts_text}

Which answer is correct based on the retrieved memories?"""

        messages = [
            {"role": "system", "content": MCQ_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self._call_api(messages, temperature=0.0, max_tokens=1024)

    def _call_api(self, messages, temperature=0.0, max_tokens=1024):
        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=body, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                })
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(2 ** attempt)
                else:
                    err_body = e.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"API error {e.code}: {err_body[:300]}")
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(1)
        raise RuntimeError("API connection failed after 3 attempts")


# ---------------------------------------------------------------------------
# 场景加载
# ---------------------------------------------------------------------------

BENCHMARK_JSON = os.path.join(_HERE, "multisession_benchmark.json")
SCENARIOS: list[dict] = []


def load_scenarios(path: Optional[str] = None) -> list[dict]:
    """从 JSON 文件加载场景。格式见 multisession_benchmark.json。"""
    global SCENARIOS
    filepath = path or BENCHMARK_JSON
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    SCENARIOS = data.get("scenarios", data if isinstance(data, list) else [])
    print(f"  加载 {len(SCENARIOS)} 个场景 ({filepath})")
    return SCENARIOS



# ---------------------------------------------------------------------------
# 评估核心
# ---------------------------------------------------------------------------

def _clean_db():
    for suffix in ["", "-wal", "-shm"]:
        p = DB_PATH + suffix
        if os.path.exists(p):
            os.remove(p)


def write_with_age(mh, content, source, field, age_days):
    """写入记忆并设置历史时间戳。"""
    mid = mh.write(content, memory_type="utterance", source=source, field=field)
    if age_days > 0:
        ago = time.time() - age_days * 86400
        mh.conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (ago, mid))
    return mid


def setup_scenario(mh: DyadCore, scenario: dict) -> dict:
    """写入场景数据并建立 contradicted 边（从 JSON edges 读取关键词对）。

    支持两种 session 格式：
      - JSON: [{"age_days": N, "field": "personal_info", "messages": [{"role": "r", "content": "c"}, ...]}, ...]
        session 级 "field" 可选，省略则使用 scenario 级 field
      - Legacy: [(age_days, [(role, content), ...]), ...]

    edges 格式: [{"old_keyword": "kw1", "new_keyword": "kw2"}, ...]
    """
    old_ids = []
    new_ids = []

    for session in scenario["sessions"]:
        if isinstance(session, dict):
            age_days = session["age_days"]
            messages = [(m["role"], m["content"]) for m in session["messages"]]
            field = session.get("field", scenario["field"])
        else:
            age_days, messages = session
            field = scenario["field"]

        for role, content in messages:
            mid = write_with_age(mh, content, role, field, age_days)
            if age_days > 0:
                old_ids.append(mid)
            else:
                new_ids.append(mid)

    # 从 JSON edges 建立 contradicted 边
    # 支持两种格式: [{"old_keyword": "x", "new_keyword": "y"}, ...] 或 [["label_old", "label_new"], ...]
    edge_pairs = []
    edges = scenario.get("edges", [])
    for edge in edges:
        if isinstance(edge, dict):
            old_kw, new_kw = edge.get("old_keyword", ""), edge.get("new_keyword", "")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            old_kw, new_kw = edge[0], edge[1]
        else:
            continue
        # Normalize: strip _old/_new suffixes from label-based keys
        old_kw = old_kw.replace("_old", "").replace("_new", "").replace("_", " ")
        new_kw = new_kw.replace("_old", "").replace("_new", "").replace("_", " ")
        if not old_kw or not new_kw:
            continue

        old_match = None
        new_match = None

        # Search across ALL messages (old + new), not just old_ids
        all_ids = old_ids + new_ids
        for mid in all_ids:
            content = mh.conn.execute(
                "SELECT content FROM memories WHERE id = ?", (mid,)
            ).fetchone()["content"]
            if old_kw.lower() in content.lower() and old_match is None:
                old_match = mid
            if new_kw.lower() in content.lower() and new_match is None:
                new_match = mid
            if old_match and new_match:
                break

        if old_match and new_match:
            mh.add_relation(new_match, old_match, "contradicted", strength=0.9)
            edge_pairs.append((old_match, new_match))

    return {"old_ids": old_ids, "new_ids": new_ids, "edge_pairs": edge_pairs}


def check_recall_ranking(mh: DyadCore, scenario: dict, info: dict) -> dict:
    """检查召回排序：新记忆是否排在旧记忆前面。"""
    results_per_q = {}

    for q in scenario["questions"]:
        question, options, correct_letter = q["question"], q["options"], q["correct"]
        q_field = q.get("field", scenario["field"])
        results = mh.recall(query=question, field_hint=q_field, limit=10)
        result_ids = [r["id"] for r in results]
        results_per_q[question[:60]] = {
            "result_ids": result_ids[:10],
            "old_in_top5": len(set(info["old_ids"]) & set(result_ids[:5])),
            "new_in_top5": len(set(info["new_ids"]) & set(result_ids[:5])),
        }

    # Overall ranking metric
    total_new_ahead = 0
    total_pairs = 0
    for old_id, new_id in info["edge_pairs"]:
        for q in scenario["questions"]:
            question = q["question"]
            q_field = q.get("field", scenario["field"])
            results = mh.recall(query=question, field_hint=q_field, limit=10)
            result_ids = [r["id"] for r in results]
            if old_id in result_ids and new_id in result_ids:
                total_pairs += 1
                if result_ids.index(new_id) < result_ids.index(old_id):
                    total_new_ahead += 1
            elif new_id in result_ids and old_id not in result_ids:
                total_new_ahead += 1
                total_pairs += 1

    return {
        "new_ahead_ratio": total_new_ahead / max(total_pairs, 1),
        "total_pairs_tested": total_pairs,
        "per_question": results_per_q,
    }


def evaluate_one_config(mh_factory, llm: Optional[LLMClient],
                        label: str) -> dict:
    """评估一个配置：拓扑渲染始终开启，记忆不随时间衰减。"""
    _clean_db()
    mh = mh_factory()

    all_results = []
    scenario_summaries = []

    for sc in SCENARIOS:
        info = setup_scenario(mh, sc)

        # 检索排序验证
        ranking = check_recall_ranking(mh, sc, info)

        sc_results = []
        for q in sc["questions"]:
            question, options, correct_letter = q["question"], q["options"], q["correct"]
            # 召回 — 支持 per-question field 覆盖
            q_field = q.get("field", sc["field"])
            results = mh.recall(query=question, field_hint=q_field, limit=5)

            # 格式化上下文 — 拓扑渲染始终开启
            memory_context = format_for_prompt(results, dyadcore=mh)

            llm_answer = ""
            correct = False
            error = ""

            if llm and not getattr(llm, '_dry_run', False):
                try:
                    llm_answer = llm.ask_mcq(question, options, memory_context)
                except Exception as e:
                    error = str(e)[:200]
            else:
                llm_answer = "(dry)"

            # 评分
            if llm_answer and not error:
                correct = _check_mcq_answer(llm_answer, correct_letter)

            sc_results.append({
                "question": question[:100],
                "correct_answer": correct_letter,
                "llm_answer": llm_answer,
                "correct": correct,
                "error": error,
                "recall_count": len(results),
            })

        correct_count = sum(1 for r in sc_results if r["correct"])
        scenario_summaries.append({
            "name": sc["name"],
            "accuracy": correct_count / max(len(sc_results), 1),
            "correct": correct_count,
            "total": len(sc_results),
            "new_ahead_ratio": ranking["new_ahead_ratio"],
            "pairs_tested": ranking["total_pairs_tested"],
            "edge_pairs": len(info["edge_pairs"]),
        })
        all_results.extend(sc_results)

    mh.close()

    total_correct = sum(1 for r in all_results if r["correct"])
    total_questions = len(all_results) or 1
    overall_accuracy = total_correct / total_questions

    avg_new_ahead = (sum(s["new_ahead_ratio"] for s in scenario_summaries)
                     / max(len(scenario_summaries), 1))

    return {
        "config": label,
        "overall_accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_questions,
        "avg_new_ahead": avg_new_ahead,
        "scenarios": scenario_summaries,
        "results": all_results,
    }


def _check_mcq_answer(llm_output: str, correct: str) -> bool:
    """检查 MCQ 答案是否正确。"""
    m = re.search(r'\(([a-d])\)', llm_output.lower())
    if m:
        given = f"({m.group(1)})"
        return given == correct.lower()
    stripped = llm_output.strip().lower()
    if stripped in ('a', 'b', 'c', 'd'):
        return f"({stripped})" == correct.lower()
    return False


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def print_report(all_configs: list[dict]):
    print(f"\n{'='*72}")
    print(f"  多会话偏好修正评估 — 消融实验结果")
    print(f"{'='*72}")

    # Header
    print(f"\n  {'Config':<36s} {'Acc':>7s}  {'New>Ahead':>9s}")
    print(f"  {'-'*56}")

    for cfg in all_configs:
        print(f"  {cfg['config']:<36s} "
              f"{cfg['overall_accuracy']:6.1%}  "
              f"{cfg['avg_new_ahead']:8.1%}")

    # Per-scenario breakdown
    print(f"\n  --- 场景分解 ---")
    for cfg in all_configs:
        print(f"  Config: {cfg['config']}")
        for sc in cfg["scenarios"]:
            print(f"  {sc['name']:<50s} {sc['accuracy']:.1%}  "
                  f"({sc['correct']}/{sc['total']})  "
                  f"new_ahead={sc['new_ahead_ratio']:.1%}  edges={sc['edge_pairs']}")

    # Detailed errors
    if all_configs:
        print(f"\n  --- 错题详情 ---")
        cfg = all_configs[0]
        for r in cfg["results"]:
            if not r["correct"] and not r.get("error"):
                print(f"  [XX] Q: {r['question'][:80]}")
                print(f"       LLM: {r['llm_answer']:<6s}  GT: {r['correct_answer']}")
            elif r.get("error"):
                print(f"  [ER] Q: {r['question'][:80]}")
                print(f"       Error: {r['error'][:100]}")

    print()


def save_results(all_configs: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="多会话偏好修正评估")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检查检索排序，不调 LLM")
    parser.add_argument("--config", type=str, default=None,
                        help="仅测试指定配置 (A/B/C/D)")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str,
                        default="multisession_results.json")
    parser.add_argument("--scenario", type=str, default=None,
                        help="仅测试指定场景名（含关键词即可）")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="场景 JSON 文件路径（默认: multisession_benchmark.json）")
    args = parser.parse_args()

    # 加载场景
    load_scenarios(args.scenarios)

    base_url = args.base_url or os.environ.get("DYADCORE_LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key = args.api_key or os.environ.get("DYADCORE_LLM_API_KEY", DEFAULT_API_KEY)
    model = args.model or os.environ.get("DYADCORE_LLM_MODEL", DEFAULT_MODEL)

    # 初始化 LLM
    llm = None
    if not args.dry_run:
        print(f"  LLM: {model} @ {base_url}")
        llm = LLMClient(base_url=base_url, api_key=api_key, model=model)
    else:
        print("  Dry run — 仅检查检索排序")
        # 创建 dummy llm 用于跳过 API 调用
        llm = type('Dummy', (), {'_dry_run': True, 'ask_mcq': lambda *a, **k: '(dry)'})()

    # 过滤场景
    global SCENARIOS
    if args.scenario:
        SCENARIOS = [s for s in SCENARIOS if args.scenario.lower() in s["name"].lower()]
        if not SCENARIOS:
            print(f"  [ERROR] 无匹配场景: {args.scenario}")
            sys.exit(1)
        print(f"  筛选场景: {SCENARIOS[0]['name']}")

    # 工厂函数 — 记忆不随时间衰减，排序仅依赖图拓扑
    def make_mh():
        return DyadCore(DB_PATH)

    # 配置矩阵 — 拓扑渲染始终开启
    all_configs_defs = [
        ("A. topo=ON  (contradicted + echoed + field strength)", make_mh),
    ]

    if args.config:
        selector = args.config.upper().strip()
        all_configs_defs = [(label, factory)
                            for label, factory in all_configs_defs
                            if label.startswith(selector)]

    all_configs = []
    for label, factory in all_configs_defs:
        print(f"\n  >>> 测试: {label}")
        result = evaluate_one_config(factory, llm, label)
        all_configs.append(result)

    print_report(all_configs)
    save_results(all_configs, args.output)


if __name__ == "__main__":
    main()
