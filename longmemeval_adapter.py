#!/usr/bin/env python3
"""
LongMemEval Adapter — DyadCore 在标准多会话记忆 benchmark 上的评估

LongMemEval 有真实时间跨度（0-304天）、多会话（38-62 个/题）、
knowledge-update 和 preference 题型直接测试偏好修正能力。

Usage:
    python longmemeval_adapter.py                      # 全量 knowledge-update + preference (108题)
    python longmemeval_adapter.py --limit 20            # 前 20 题
    python longmemeval_adapter.py --question-type knowledge-update
    python longmemeval_adapter.py --dry-run             # 只看检索不看 LLM

环境变量：
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
DATASET_NAME = "xiaowu0162/longmemeval-cleaned"
SPLIT = "longmemeval_s_cleaned"
DB_PATH = "longmemeval_test.db"

# 每种题型 → DyadCore field 映射
QTYPE_FIELD_MAP = {
    "knowledge-update": "project_tech",
    "single-session-preference": "personal_info",
    "multi-session": "meeting_notes",
    "temporal-reasoning": "meeting_notes",
    "single-session-user": "personal_info",
    "single-session-assistant": "personal_info",
}

ANSWER_SYSTEM_PROMPT = """You are an assistant that answers questions based on retrieved user memories.

Rules:
1. Read the retrieved memories carefully
2. Answer the question based ONLY on what the memories contain
3. Be concise — answer in 1-3 sentences
4. If the memories don't contain the answer, say "I don't have enough information"
5. Do not make up facts not present in the memories"""

# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY,
                 model=DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def answer_question(self, question: str, memory_context: str) -> str:
        prompt = f"""## Retrieved User Memories

{memory_context}

## Question

{question}

Answer the question based on the retrieved memories above."""

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self._call_api(messages, temperature=0.0, max_tokens=256)

    def judge_answer(self, question: str, ground_truth: str,
                     llm_answer: str) -> bool:
        """LLM-as-judge: does llm_answer match ground_truth for the question?"""
        prompt = f"""## Question
{question}

## Ground Truth Answer
{ground_truth}

## Generated Answer
{llm_answer}

Does the Generated Answer contain the same key facts as the Ground Truth Answer?
Answer ONLY "yes" or "no"."""

        messages = [
            {"role": "system", "content": "You are an evaluator. Compare answers for factual match."},
            {"role": "user", "content": prompt},
        ]
        response = self._call_api(messages, temperature=0.0, max_tokens=512)
        return "yes" in response.lower()

    def _call_api(self, messages, temperature=0.0, max_tokens=512):
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
                    content = data["choices"][0]["message"]["content"]
                    return (content or "").strip()
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
# Dataset loader
# ---------------------------------------------------------------------------
def load_longmemeval(question_types: Optional[list[str]] = None,
                     limit: Optional[int] = None) -> list[dict]:
    """Load LongMemEval S split via streaming, return filtered examples."""
    from datasets import load_dataset

    print(f"Loading {DATASET_NAME} ({SPLIT}) via streaming...")
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    examples = []
    for item in ds:
        qt = item["question_type"]
        if question_types and qt not in question_types:
            continue
        examples.append(item)
        if limit and len(examples) >= limit:
            break

    print(f"  Loaded {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class LongMemEvalAdapter:
    def __init__(self, llm: LLMClient, db_path: str = DB_PATH,
                 use_judge: bool = True):
        self.llm = llm
        self.db_path = db_path
        self.use_judge = use_judge
        self.mh: Optional[DyadCore] = None
        self._results: list[dict] = []

    def evaluate(self, examples: list[dict], dry_run: bool = False) -> dict:
        """Run evaluation on all examples."""
        total = len(examples)
        correct_keyword = 0
        correct_judge = 0
        skipped = 0
        results: list[dict] = []

        for idx, example in enumerate(examples):
            # Initialize fresh DyadCore per example
            self._init_dyadcore()

            # Parse and write all sessions
            self._write_sessions(example)

            # Answer question
            result = self._answer_one(example, dry_run)
            results.append(result)

            if result.get("keyword_correct"):
                correct_keyword += 1
            if result.get("judge_correct"):
                correct_judge += 1
            if result.get("skipped"):
                skipped += 1

            # Progress
            done = len(results)
            if done % 10 == 0 or done == total:
                acc = correct_keyword / max(done - skipped, 1)
                print(f"  [{done}/{total}] kw_acc={acc:.1%} "
                      f"({correct_keyword}/{done - skipped}), skipped={skipped}")

            self.mh.close()

        metrics = self._compute_metrics(results)
        self._results = results
        return metrics

    def _init_dyadcore(self):
        for suffix in ["", "-wal", "-shm"]:
            p = self.db_path + suffix
            if os.path.exists(p):
                os.remove(p)
        self.mh = DyadCore(self.db_path)

    def _write_sessions(self, example: dict):
        """Write all haystack sessions to DyadCore with timestamps."""
        dates = example.get("haystack_dates", [])
        session_ids = example.get("haystack_session_ids", [])
        sessions = example.get("haystack_sessions", [])

        field = QTYPE_FIELD_MAP.get(example.get("question_type", ""), "personal_info")

        for i, session in enumerate(sessions):
            # Parse session messages (may be JSON strings or dicts)
            messages = self._parse_session(session)
            if not messages:
                continue

            # Get timestamp for this session
            timestamp = None
            if i < len(dates):
                try:
                    from datetime import datetime
                    dt = datetime.strptime(dates[i], "%Y/%m/%d (%a) %H:%M")
                    timestamp = dt.timestamp()
                except (ValueError, TypeError):
                    pass

            for msg in messages:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue

                # Normalize role: LongMemEval uses "assistant", DyadCore uses "agent"
                if role == "assistant":
                    role = "agent"
                elif role not in ("user", "agent"):
                    continue

                mid = self.mh.write(content, memory_type="utterance",
                                   source=role, field=field)

                # Set historical timestamp
                if timestamp and self.mh:
                    self.mh.conn.execute(
                        "UPDATE memories SET created_at = ? WHERE id = ?",
                        (timestamp, mid)
                    )

    @staticmethod
    def _parse_session(session) -> list[dict]:
        """Parse a session (list of JSON strings or dicts) into message dicts."""
        messages = []
        if not isinstance(session, list):
            return messages
        for turn in session:
            if isinstance(turn, dict):
                messages.append(turn)
            elif isinstance(turn, str):
                try:
                    parsed = json.loads(turn)
                    if isinstance(parsed, dict):
                        messages.append(parsed)
                except json.JSONDecodeError:
                    pass
        return messages

    def _answer_one(self, example: dict, dry_run: bool) -> dict:
        """Answer a single question using DyadCore recall."""
        question = example["question"]
        question_type = example["question_type"]
        ground_truth = example.get("answer", "").strip()
        field = QTYPE_FIELD_MAP.get(question_type, "personal_info")

        # Retrieve memories
        results = self.mh.recall(query=question, field_hint=field, limit=5) if self.mh else []
        memory_context = format_for_prompt(results, dyadcore=self.dc)

        if dry_run:
            # Check if key terms from ground truth appear in any retrieved memory
            gt_keywords = self._extract_keywords(ground_truth)
            recall_hit = any(
                sum(1 for kw in gt_keywords if kw.lower() in r.get("content", "").lower())
                >= min(2, len(gt_keywords) // 2 + 1)
                for r in results
            ) if gt_keywords else False
            return {
                "question_id": example.get("question_id", ""),
                "question_type": question_type,
                "question": question[:120],
                "ground_truth": ground_truth[:150],
                "llm_answer": "(dry)",
                "keyword_correct": False,
                "judge_correct": False,
                "recall_hit": recall_hit,
                "recall_count": len(results),
                "skipped": False,
            }

        # Real LLM call
        try:
            llm_answer = self.llm.answer_question(question, memory_context)
        except Exception as e:
            return {
                "question_id": example.get("question_id", ""),
                "question_type": question_type,
                "question": question[:120],
                "ground_truth": ground_truth[:150],
                "llm_answer": "",
                "keyword_correct": False,
                "judge_correct": False,
                "recall_count": len(results),
                "error": str(e)[:200],
                "skipped": False,
            }

        # Score: keyword overlap
        kw_correct = self._check_keyword_match(llm_answer, ground_truth)

        # Score: LLM judge (optional)
        judge_correct = False
        if self.use_judge and llm_answer:
            try:
                judge_correct = self.llm.judge_answer(question, ground_truth, llm_answer)
            except Exception:
                judge_correct = kw_correct  # fallback to keyword

        return {
            "question_id": example.get("question_id", ""),
            "question_type": question_type,
            "question": question[:120],
            "ground_truth": ground_truth[:150],
            "llm_answer": llm_answer[:300],
            "keyword_correct": kw_correct,
            "judge_correct": judge_correct,
            "recall_count": len(results),
            "error": "",
            "skipped": False,
        }

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from answer text."""
        # Remove quotes, split on common delimiters
        text = re.sub(r'["\']', '', text)
        # Get words >= 4 chars
        words = [w for w in re.findall(r'[a-zA-Z]{4,}', text.lower())
                 if w not in ('that', 'this', 'with', 'from', 'have', 'been',
                              'were', 'they', 'them', 'your', 'what', 'when',
                              'where', 'which', 'there', 'their', 'would', 'could',
                              'should', 'about', 'which', 'than', 'also', 'some',
                              'only', 'very', 'just', 'like', 'more', 'will')]
        return list(set(words))[:10]

    @staticmethod
    def _check_keyword_match(llm_answer: str, ground_truth: str) -> bool:
        """Check if key terms from ground truth appear in LLM answer."""
        if not llm_answer or not ground_truth:
            return False
        keywords = LongMemEvalAdapter._extract_keywords(ground_truth)
        if not keywords:
            return False
        hits = sum(1 for kw in keywords if kw.lower() in llm_answer.lower())
        return hits >= min(2, len(keywords) // 3 + 1)

    def _compute_metrics(self, results: list[dict]) -> dict:
        valid = [r for r in results if not r.get("error")]
        errors = [r for r in results if r.get("error")]

        kw_correct = sum(1 for r in valid if r.get("keyword_correct"))
        judge_correct = sum(1 for r in valid if r.get("judge_correct"))
        total = len(valid) or 1

        by_type = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in valid:
            qt = r.get("question_type", "unknown")
            by_type[qt]["total"] += 1
            if r.get("keyword_correct"):
                by_type[qt]["correct"] += 1

        return {
            "keyword_accuracy": kw_correct / total,
            "judge_accuracy": judge_correct / total,
            "total_questions": len(results),
            "valid_answers": total,
            "errors": len(errors),
            "by_question_type": {
                qt: {"accuracy": d["correct"] / max(d["total"], 1),
                     "correct": d["correct"], "total": d["total"]}
                for qt, d in sorted(by_type.items())
            },
        }

    def save_results(self, path: str):
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
    parser = argparse.ArgumentParser(description="LongMemEval DyadCore Adapter")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-type", type=str, default=None,
                        help="Filter: knowledge-update, single-session-preference, etc.")
    parser.add_argument("--question-types", type=str, default=None,
                        help="Comma-separated: knowledge-update,single-session-preference")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (keyword scoring only)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str,
                        default="longmemeval_results.json")
    args = parser.parse_args()

    base_url = args.base_url or os.environ.get("DYADCORE_LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key = args.api_key or os.environ.get("DYADCORE_LLM_API_KEY", DEFAULT_API_KEY)
    model = args.model or os.environ.get("DYADCORE_LLM_MODEL", DEFAULT_MODEL)

    # Determine question types
    if args.question_types:
        question_types = [t.strip() for t in args.question_types.split(",")]
    elif args.question_type:
        question_types = [args.question_type]
    else:
        question_types = ["knowledge-update", "single-session-preference"]

    # Load data
    examples = load_longmemeval(question_types=question_types, limit=args.limit)

    # Init LLM
    llm = None
    if not args.dry_run:
        print(f"\n  LLM: {model} @ {base_url}")
        llm = LLMClient(base_url=base_url, api_key=api_key, model=model)

    # Evaluate
    print(f"  Question types: {question_types}")
    print(f"  LLM Judge: {not args.no_judge}")
    print()

    adapter = LongMemEvalAdapter(
        llm,
        use_judge=not args.no_judge and not args.dry_run,
    )
    metrics = adapter.evaluate(examples, dry_run=args.dry_run)

    # Report
    print(f"\n{'='*64}")
    print(f"  LongMemEval Results")
    print(f"{'='*64}")
    print(f"  Keyword Accuracy: {metrics['keyword_accuracy']:.1%} "
          f"({sum(1 for r in adapter._results if r.get('keyword_correct'))}/{metrics['valid_answers']})")
    if metrics.get('judge_accuracy', 0) > 0:
        print(f"  Judge Accuracy:   {metrics['judge_accuracy']:.1%}")
    if metrics['errors']:
        print(f"  Errors: {metrics['errors']}")

    print(f"\n  By Question Type:")
    for qt, d in metrics["by_question_type"].items():
        print(f"  {qt:35s} {d['accuracy']:6.1%}  {d['correct']:3d}/{d['total']}")

    adapter.save_results(args.output)


if __name__ == "__main__":
    main()
