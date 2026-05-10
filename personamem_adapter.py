#!/usr/bin/env python3
"""
PersonaMem Adapter — DyadCore 在标准 benchmark 上的评估

将 PersonaMem 的对话历史写入 DyadCore，用 recall + LLM 回答多选题，
对比 ground truth 计算准确率。

Usage:
    python personamem_adapter.py                    # 全量 (589 题)
    python personamem_adapter.py --limit 50         # 前 50 题
    python personamem_adapter.py --question-type recall_user_shared_facts  # 仅某类题
    python personamem_adapter.py --dry-run           # 只看数据不调 LLM

环境变量（同 eval_conversation.py）：
    DYADCORE_LLM_BASE_URL / DYADCORE_LLM_API_KEY / DYADCORE_LLM_MODEL
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# 路径 & 导入
# ---------------------------------------------------------------------------
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
DATASET_NAME = "bowen-upenn/PersonaMem"
SPLIT = "32k"

# 各 topic → DyadCore field 映射
TOPIC_FIELD_MAP = {
    "movieRecommendation": "personal_info",
    "musicRecommendation": "personal_info",
    "bookRecommendation": "personal_info",
    "foodRecommendation": "personal_info",
    "travelPlanning": "personal_info",
    "datingConsultation": "personal_info",
    "familyRelations": "personal_info",
    "studyConsultation": "personal_info",
    "homeDecoration": "personal_info",
    "financialConsultation": "project_tech",
    "medicalConsultation": "meeting_notes",
    "therapy": "meeting_notes",
    "legalConsultation": "project_tech",
}

MCQ_SYSTEM_PROMPT = """You are an assistant that answers multiple-choice questions based on retrieved user memories.

Rules:
1. Read the retrieved memories carefully — they contain the user's preferences, history, and traits
2. Pick the SINGLE best answer (a, b, c, or d) that matches what the user has said
3. Output ONLY the letter in parentheses, like "(a)" or "(b)"
4. Do not explain, do not add any other text — just the answer letter

If the memories don't contain enough information to answer, make your best guess based on what IS available."""


# ---------------------------------------------------------------------------
# LLM Client (same as eval_conversation.py)
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 api_key: str = DEFAULT_API_KEY,
                 model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def ask(self, question: str, options_text: str, memory_context: str) -> str:
        """Ask LLM to answer a multiple-choice question."""
        prompt = f"""## Retrieved User Memories

{memory_context}

## Question

{question}

## Options

{options_text}

Which answer is correct based on the retrieved memories?"""

        messages = [
            {"role": "system", "content": MCQ_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self._call_api(messages, temperature=0.0, max_tokens=1024)

    def _call_api(self, messages: list[dict], temperature: float = 0.0,
                  max_tokens: int = 512) -> str:
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
                with urllib.request.urlopen(req, timeout=60) as resp:
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
        raise RuntimeError(f"API connection failed after 3 attempts")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_personamem():
    """Load PersonaMem 32k split: questions + shared_contexts."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print(f"Loading {DATASET_NAME} ({SPLIT} split)...")
    questions = load_dataset(DATASET_NAME, split=SPLIT)

    # Download shared_contexts
    ctx_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=f"shared_contexts_{SPLIT}.jsonl",
        repo_type="dataset"
    )

    # Parse JSONL: {shared_context_id: [messages]}
    contexts = {}
    with open(ctx_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            contexts.update(obj)

    print(f"  Questions: {len(questions)}")
    print(f"  Shared contexts: {len(contexts)}")
    return questions, contexts


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class PersonaMemAdapter:
    def __init__(self, llm: LLMClient, dyadcore_db: str = "personamem_test.db"):
        self.llm = llm
        self.db_path = dyadcore_db
        self.mh: Optional[DyadCore] = None
        self._results: list[dict] = []

    def evaluate(self, questions, contexts, limit: Optional[int] = None,
                 question_type_filter: Optional[str] = None,
                 dry_run: bool = False) -> dict:
        """Run evaluation.

        Strategy: 按 shared_context_id 分组，每组的消息写入 DyadCore，
        然后回答该组所有问题。组间清除数据库。
        """
        # 过滤
        qs = list(questions)
        if question_type_filter:
            qs = [q for q in qs if q["question_type"] == question_type_filter]
        if limit:
            qs = qs[:limit]

        # 按 shared_context_id 分组
        groups = defaultdict(list)
        for q in qs:
            groups[q["shared_context_id"]].append(q)

        print(f"\n  评估 {len(qs)} 道题（{len(groups)} 个上下文组）")
        print(f"  Dry run: {dry_run}")
        print()

        total = len(qs)
        correct = 0
        skipped = 0
        results: list[dict] = []

        ctx_count = 0
        for ctx_id, group_qs in groups.items():
            ctx_count += 1
            if ctx_id not in contexts:
                print(f"  [WARN] shared_context_id {ctx_id[:16]}... not found in contexts, skipping {len(group_qs)} qs")
                skipped += len(group_qs)
                continue

            messages = contexts[ctx_id]
            # 按 end_index 排序（增量写入）
            group_qs.sort(key=lambda q: q["end_index_in_shared_context"])

            # 重新初始化 DyadCore（每个上下文组独立）
            self._init_dyadcore()

            last_end = 0
            for q in group_qs:
                end_idx = q["end_index_in_shared_context"]
                # 写入新消息（从上一次 end 到当前 end）
                if end_idx > last_end:
                    self._write_messages(messages[last_end:end_idx], q.get("topic", ""))
                    last_end = end_idx

                # 回答
                result = self._answer_one(q, dry_run)
                results.append(result)

                if result.get("correct"):
                    correct += 1

                # 进度（每 10 题输出一次）
                done = len(results)
                if done % 10 == 0 or done == len(qs):
                    acc = correct / max(done - skipped, 1)
                    print(f"  [{done}/{len(qs)}] acc={acc:.1%} "
                          f"({correct}/{done - skipped} correct, {skipped} skipped)")

            self.mh.close()

        # 汇总
        metrics = self._compute_metrics(results)
        self._results = results
        return metrics

    def _init_dyadcore(self):
        """Initialize fresh DyadCore instance."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        for ext in ["-wal", "-shm"]:
            p = self.db_path + ext
            if os.path.exists(p):
                os.remove(p)
        self.mh = DyadCore(self.db_path)

    def _write_messages(self, messages: list[dict], topic: str):
        """Write conversation messages as memories.

        system prompt → agent meta (persona description)
        user → user utterance
        assistant → agent utterance
        """
        field = TOPIC_FIELD_MAP.get(topic, topic)
        if not self.mh:
            return

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue

            if role == "system":
                # Extract key persona traits from system prompt
                # These are long — write as agent meta
                self.mh.write(content, memory_type="meta", source="agent", field=field)
            elif role == "user":
                self.mh.write(content, memory_type="utterance", source="user", field=field)
            elif role == "assistant":
                self.mh.write(content, memory_type="utterance", source="agent", field=field)

    def _answer_one(self, q: dict, dry_run: bool) -> dict:
        """Answer a single question using DyadCore recall."""
        question_text = q["user_question_or_message"]
        topic = q.get("topic", "")
        field = TOPIC_FIELD_MAP.get(topic, topic)
        correct_answer = (q.get("correct_answer") or "").strip()
        options_text = q.get("all_options", "")
        question_type = q.get("question_type", "")

        if dry_run:
            # Check recall coverage — does retrieved memory overlap with correct answer text?
            results = self.mh.recall(query=question_text, field_hint=field, limit=5) if self.mh else []
            recall_hit = correct_answer and any(
                self._answer_in_memory(correct_answer, r.get("content", ""), options_text)
                for r in results
            )
            return {
                "question_id": q.get("question_id", ""),
                "question_type": question_type,
                "topic": topic,
                "question": question_text[:120],
                "correct_answer": correct_answer,
                "llm_answer": "(dry)",
                "correct": False,
                "recall_hit": recall_hit,
                "recall_count": len(results),
                "error": "",
            }

        # Real LLM call
        memory_context = ""
        if self.mh:
            results = self.mh.recall(query=question_text, field_hint=field, limit=5)
            if results:
                memory_context = format_for_prompt(results, dyadcore=self.dc)

        try:
            llm_answer = self.llm.ask(question_text, options_text, memory_context)
        except Exception as e:
            return {
                "question_id": q.get("question_id", ""),
                "question_type": question_type,
                "topic": topic,
                "question": question_text[:120],
                "correct_answer": correct_answer,
                "llm_answer": "",
                "correct": False,
                "error": str(e)[:200],
            }

        correct = self._check_answer(llm_answer, correct_answer)

        return {
            "question_id": q.get("question_id", ""),
            "question_type": question_type,
            "topic": topic,
            "question": question_text[:120],
            "correct_answer": correct_answer,
            "llm_answer": llm_answer,
            "correct": correct,
            "error": "",
        }

    @staticmethod
    def _check_answer(llm_output: str, correct: str) -> bool:
        """Check if LLM answer matches ground truth."""
        # Extract (a), (b), (c), or (d) from output
        m = re.search(r'\(([a-d])\)', llm_output.lower())
        if m:
            given = f"({m.group(1)})"
            return given == correct.lower()
        # Try bare letter
        stripped = llm_output.strip().lower()
        if stripped in ('a', 'b', 'c', 'd'):
            return f"({stripped})" == correct.lower()
        return False

    @staticmethod
    def _answer_in_memory(correct_answer: str, content: str, correct_option_text: str = "") -> bool:
        """Check if the memory content overlaps with the correct answer option text."""
        # Extract key terms from correct option (words >= 4 chars, skip common words)
        if correct_option_text:
            # Remove the answer letter prefix like "(c)"
            text = re.sub(r'^\([a-d]\)\s*', '', correct_option_text)
            words = [w for w in re.findall(r'[a-zA-Z]{4,}', text.lower())
                     if w not in ('that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'them', 'your', 'what', 'when', 'where', 'which', 'there', 'their')]
            if words:
                hits = sum(1 for w in words if w in content.lower())
                return hits >= min(2, len(words) // 3 + 1)
        return False

    def _compute_metrics(self, results: list[dict]) -> dict:
        """Compute accuracy metrics by question type."""
        valid = [r for r in results if not r.get("error")]
        errors = [r for r in results if r.get("error")]
        correct = sum(1 for r in valid if r.get("correct"))
        total = len(valid) or 1

        by_type = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in valid:
            qt = r.get("question_type", "unknown")
            by_type[qt]["total"] += 1
            if r.get("correct"):
                by_type[qt]["correct"] += 1

        by_topic = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in valid:
            t = r.get("topic", "unknown")
            by_topic[t]["total"] += 1
            if r.get("correct"):
                by_topic[t]["correct"] += 1

        return {
            "overall_accuracy": correct / max(total, 1),
            "total_questions": len(results),
            "valid_answers": total,
            "correct_answers": correct,
            "errors": len(errors),
            "by_question_type": {
                qt: {"accuracy": d["correct"] / max(d["total"], 1),
                      "correct": d["correct"], "total": d["total"]}
                for qt, d in sorted(by_type.items())
            },
            "by_topic": {
                t: {"accuracy": d["correct"] / max(d["total"], 1),
                     "correct": d["correct"], "total": d["total"]}
                for t, d in sorted(by_topic.items())
            },
        }

    def save_results(self, path: str):
        """Save full evaluation results to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": self._compute_metrics(self._results),
                "results": self._results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Results saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PersonaMem DyadCore Adapter")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--question-type", type=str, default=None,
                        help="Filter by question type")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check recall, don't call LLM")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str,
                        default="personamem_results.json")
    args = parser.parse_args()

    base_url = args.base_url or os.environ.get("DYADCORE_LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key = args.api_key or os.environ.get("DYADCORE_LLM_API_KEY", DEFAULT_API_KEY)
    model = args.model or os.environ.get("DYADCORE_LLM_MODEL", DEFAULT_MODEL)

    # Load data
    questions, contexts = load_personamem()

    # Init LLM (skip if dry run)
    llm = None
    if not args.dry_run:
        print(f"\n  LLM: {model} @ {base_url}")
        llm = LLMClient(base_url=base_url, api_key=api_key, model=model)

    # Evaluate
    adapter = PersonaMemAdapter(llm)
    metrics = adapter.evaluate(
        questions, contexts,
        limit=args.limit,
        question_type_filter=args.question_type,
        dry_run=args.dry_run,
    )

    # Report
    print(f"\n{'='*64}")
    print(f"  PersonaMem 评估结果")
    print(f"{'='*64}")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.1%} "
          f"({metrics['correct_answers']}/{metrics['valid_answers']})")
    if metrics['errors']:
        print(f"  Errors: {metrics['errors']}")
    print(f"\n  By Question Type:")
    print(f"  {'Type':50s} {'Acc':>6s}  {'Correct':>7s}")
    print(f"  {'-'*65}")
    for qt, d in metrics["by_question_type"].items():
        print(f"  {qt:50s} {d['accuracy']:6.1%}  {d['correct']:4d}/{d['total']:<4d}")

    print(f"\n  By Topic:")
    print(f"  {'Topic':30s} {'Acc':>6s}  {'Correct':>7s}")
    print(f"  {'-'*45}")
    for t, d in metrics["by_topic"].items():
        print(f"  {t:30s} {d['accuracy']:6.1%}  {d['correct']:4d}/{d['total']:<4d}")

    adapter.save_results(args.output)


if __name__ == "__main__":
    main()
