
"""
Operational evals, merged: LATENCY, COST, RELIABILITY.

Measures whether the RAG app runs quickly, economically, and reliably.
The generation LLM is configured in src/generator.py.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import math
import time
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import generate, generate_stream, prompt, llm

load_dotenv()

# Preserve the AIMessage to read token usage metadata.
measured_chain = prompt | llm


# ============================================================
# 2. SHARED CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]


# ============================================================
# 3. SHARED HELPERS
# ============================================================
def percentile(values, p):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")

    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)

    if lo == hi:
        return s[int(k)]

    return s[lo] * (hi - k) + s[hi] * (k - lo)


def col_avg(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


# ============================================================================
# LATENCY
# ============================================================================

LAT_REPEATS = 5
LAT_WARMUP_RUNS = 2
LAT_MEASURE_TTFT = True
LAT_STAGE_LEVEL = True

SLO_P95_MS = 3000
SLO_TTFT_P95_MS = 1200


def lat_end_to_end(pipeline, question):
    return pipeline.invoke(question)["answer"]


def lat_stages(pipeline, question):
    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    t1 = time.perf_counter()
    answer = generate(question, context)
    t2 = time.perf_counter()

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
    }


def lat_stages_streaming(pipeline, question):
    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    t1 = time.perf_counter()

    first_token_t = None
    pieces = []

    for piece in generate_stream(question, context):
        if piece:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            pieces.append(piece)

    t2 = time.perf_counter()

    answer = "".join(pieces)
    ttft_ms = (
        (first_token_t - t0) * 1000
        if first_token_t is not None
        else float("nan")
    )

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }


def lat_benchmark(pipeline):
    print(f"[latency] warming up ({LAT_WARMUP_RUNS} runs, discarded)...")

    for i in range(LAT_WARMUP_RUNS):
        lat_end_to_end(pipeline, QUESTIONS[i % len(QUESTIONS)])

    total_ms = []
    retrieval_ms = []
    generation_ms = []
    ttft_ms = []
    answer_lengths = []

    print("[latency] measuring...")

    for question in QUESTIONS:
        for _ in range(LAT_REPEATS):
            start = time.perf_counter()

            if LAT_MEASURE_TTFT:
                answer, stage = lat_stages_streaming(pipeline, question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
                ttft_ms.append(stage["ttft"])

            elif LAT_STAGE_LEVEL:
                answer, stage = lat_stages(pipeline, question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])

            else:
                answer = lat_end_to_end(pipeline, question)

            elapsed_ms = (time.perf_counter() - start) * 1000

            total_ms.append(elapsed_ms)
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }


def lat_summarize(samples):
    clean = [s for s in samples if not math.isnan(s)]

    if not clean:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def lat_print_row(label, s):
    print(
        f"{label:<12} | n={s['n']:<3} "
        f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
        f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
        f"min={s['min']:7.1f}  max={s['max']:7.1f}"
    )


def lat_slo_line(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"

    print(
        f"SLO: {label:<22} p95 <= {budget:>5} ms  "
        f"->  p95 = {p95:7.0f} ms   [{verdict}]"
    )


def lat_report(results):
    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)

    print(
        f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} "
        f"{'p95':>11} {'p99':>11} {'min':>11} {'max':>11}"
    )
    print("-" * 78)

    total = lat_summarize(results["total"])
    lat_print_row("end-to-end", total)

    if results["ttft"]:
        lat_print_row("ttft", lat_summarize(results["ttft"]))

    if results["retrieval"]:
        lat_print_row("retrieval", lat_summarize(results["retrieval"]))
        lat_print_row("generation", lat_summarize(results["generation"]))

    avg_len = (
        sum(results["answer_len"]) / len(results["answer_len"])
        if results["answer_len"]
        else 0
    )

    print("-" * 78)
    print(f"avg answer length: {avg_len:.0f} chars")

    print("=" * 78)

    lat_slo_line("full answer", total["p95"], SLO_P95_MS)

    if results["ttft"]:
        lat_slo_line(
            "first token (perceived)",
            lat_summarize(results["ttft"])["p95"],
            SLO_TTFT_P95_MS,
        )

    print("=" * 78)


def run_latency(pipeline, verbose=True):
    results = lat_benchmark(pipeline)

    if verbose:
        lat_report(results)

    total = lat_summarize(results["total"])

    metrics = {
        "e2e_mean_ms": total["mean"],
        "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"],
        "e2e_p99_ms": total["p99"],
        "avg_answer_len": col_avg(
            [{"length": x} for x in results["answer_len"]],
            "length",
        ),
        "slo_e2e_pass": total["p95"] <= SLO_P95_MS,
    }

    if results["ttft"]:
        ttft = lat_summarize(results["ttft"])
        metrics["ttft_p95_ms"] = ttft["p95"]
        metrics["slo_ttft_pass"] = ttft["p95"] <= SLO_TTFT_P95_MS

    if results["retrieval"]:
        metrics["retrieval_p95_ms"] = lat_summarize(
            results["retrieval"]
        )["p95"]

        metrics["generation_p95_ms"] = lat_summarize(
            results["generation"]
        )["p95"]

    return metrics


# ============================================================================
# COST — GEMINI
# ============================================================================

COST_REPEATS = 3

# Gemini 2.5 Flash standard paid-tier pricing, USD per 1M text tokens.
# These are estimates for the paid tier, not actual free-tier charges.
PRICE_INPUT_PER_1M = 0.30
PRICE_CACHED_INPUT_PER_1M = 0.03
PRICE_OUTPUT_PER_1M = 2.50

QUERIES_PER_DAY = 2000
USD_TO_INR = 88.0

COST_BUDGET_PER_QUERY_USD = 0.0015


def cost_measure_tokens(pipeline, question):
    docs = pipeline.retriever.invoke(question)
    context_text = "\n\n".join(
        doc.page_content for doc in docs
    )

    msg = measured_chain.invoke({
        "question": question,
        "context": context_text,
    })

    usage = getattr(msg, "usage_metadata", None) or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    details = usage.get("input_token_details") or {}

    cached_tokens = (
        details.get("cache_read", 0)
        or details.get("cache_read_input_tokens", 0)
        or 0
    )

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }


def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input = max(input_tokens - cached_tokens, 0)

    c_in = (
        uncached_input / 1_000_000
        * PRICE_INPUT_PER_1M
    )

    c_cached = (
        cached_tokens / 1_000_000
        * PRICE_CACHED_INPUT_PER_1M
    )

    c_out = (
        output_tokens / 1_000_000
        * PRICE_OUTPUT_PER_1M
    )

    return {
        "input": c_in,
        "cached": c_cached,
        "output": c_out,
        "total": c_in + c_cached + c_out,
    }


def cost_benchmark(pipeline):
    rows = []

    print("[cost] measuring Gemini token usage...")

    for question in QUESTIONS:
        for _ in range(COST_REPEATS):
            tok = cost_measure_tokens(pipeline, question)

            cost = cost_usd(
                tok["input"],
                tok["output"],
                tok["cached"],
            )

            rows.append({
                **tok,
                **{
                    f"cost_{k}": v
                    for k, v in cost.items()
                },
            })

    return rows


def cost_report(rows):
    if not rows:
        print("[cost] no measurements available")
        return

    n = len(rows)

    avg_in = col_avg(rows, "input")
    avg_out = col_avg(rows, "output")
    avg_cached = col_avg(rows, "cached")
    avg_cost = col_avg(rows, "cost_total")

    min_cost = min(r["cost_total"] for r in rows)
    max_cost = max(r["cost_total"] for r in rows)

    avg_cost_out = col_avg(rows, "cost_output")

    out_share = (
        100 * avg_cost_out / avg_cost
        if avg_cost
        else 0
    )

    print("\n" + "=" * 70)
    print("COST (Gemini 2.5 Flash equivalent paid-tier estimate)")
    print("=" * 70)

    print(f"samples                : {n}")
    print(f"avg input tokens       : {avg_in:8.0f} ({avg_cached:.0f} cached)")
    print(f"avg output tokens      : {avg_out:8.0f}")

    print("-" * 70)

    print(
        f"estimated cost / query: ${avg_cost:.6f} "
        f"(Rs {avg_cost * USD_TO_INR:.4f})"
    )

    print(
        f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}"
    )

    print(
        f"   input vs output     : "
        f"{100 - out_share:.0f}% input / {out_share:.0f}% output"
    )

    print("-" * 70)

    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30

    print(f"projection @ {QUERIES_PER_DAY}/day:")
    print(f"   per day             : ${daily:8.2f} (Rs {daily * USD_TO_INR:8.2f})")
    print(f"   per month           : ${monthly:8.2f} (Rs {monthly * USD_TO_INR:8.2f})")

    print("=" * 70)

    verdict = (
        "PASS"
        if avg_cost <= COST_BUDGET_PER_QUERY_USD
        else "FAIL"
    )

    print(
        f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f} "
        f"-> ${avg_cost:.6f} [{verdict}]"
    )

    print("=" * 70)
    print(
        "Note: This is an equivalent paid-tier estimate. "
        "Qualifying free-tier usage may incur no API charge."
    )


def run_cost(pipeline, verbose=True):
    rows = cost_benchmark(pipeline)

    if verbose:
        cost_report(rows)

    avg_cost = col_avg(rows, "cost_total")
    avg_cost_out = col_avg(rows, "cost_output")

    out_share = (
        100 * avg_cost_out / avg_cost
        if avg_cost
        else 0.0
    )

    return {
        "cost_per_query_usd": avg_cost,
        "cost_per_query_inr": avg_cost * USD_TO_INR,
        "avg_input_tokens": col_avg(rows, "input"),
        "avg_output_tokens": col_avg(rows, "output"),
        "avg_cached_tokens": col_avg(rows, "cached"),
        "output_cost_share_pct": out_share,
        "monthly_usd": avg_cost * QUERIES_PER_DAY * 30,
        "budget_pass": avg_cost <= COST_BUDGET_PER_QUERY_USD,
    }


# ============================================================================
# RELIABILITY
# ============================================================================

REL_REPEATS = 5
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


class Reliability:
    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0


def call_with_retries(fn, reliability):
    reliability.calls += 1

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = fn()
            reliability.successes += 1
            return result

        except Exception as e:
            if attempt < MAX_RETRIES:
                reliability.retries += 1
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))
            else:
                reliability.failures += 1
                print(
                    f"[reliability] FAILED after {MAX_RETRIES} retries: {e}"
                )
                return None


def rel_benchmark(pipeline):
    reliability = Reliability()

    print("[reliability] measuring...")

    for question in QUESTIONS:
        for _ in range(REL_REPEATS):
            call_with_retries(
                lambda q=question: pipeline.invoke(q),
                reliability,
            )

    return reliability


def rel_report(rel):
    calls = rel.calls or 1

    success_rate = 100 * rel.successes / calls
    error_rate = 100 * rel.failures / calls
    retry_rate = 100 * rel.retries / calls

    print("\n" + "=" * 60)
    print("RELIABILITY")
    print("=" * 60)

    print(f"total requests : {rel.calls}")
    print(f"successful     : {rel.successes}")
    print(f"failed         : {rel.failures}")

    print("-" * 60)

    print(f"success rate   : {success_rate:.2f}%")
    print(f"error rate     : {error_rate:.2f}%")
    print(f"retry rate     : {retry_rate:.2f}%")

    print("=" * 60)


def run_reliability(pipeline, verbose=True):
    rel = rel_benchmark(pipeline)

    if verbose:
        rel_report(rel)

    calls = rel.calls or 1

    return {
        "total_requests": rel.calls,
        "success_rate": 100 * rel.successes / calls,
        "error_rate": 100 * rel.failures / calls,
        "retry_rate": 100 * rel.retries / calls,
    }


# ============================================================================
# ENTRYPOINT — MERGED OPERATIONAL SNAPSHOT
# ============================================================================

def run_ops(pipeline=None, verbose=True):
    pipeline = pipeline or RagPipeline()

    latency = run_latency(pipeline, verbose=verbose)
    cost = run_cost(pipeline, verbose=verbose)
    reliability = run_reliability(pipeline, verbose=verbose)

    snapshot = {}

    snapshot.update({
        f"latency.{k}": v
        for k, v in latency.items()
    })

    snapshot.update({
        f"cost.{k}": v
        for k, v in cost.items()
    })

    snapshot.update({
        f"reliability.{k}": v
        for k, v in reliability.items()
    })

    return snapshot


def main():
    snapshot = run_ops(verbose=True)

    print("\n" + "=" * 70)
    print("OPERATIONAL SNAPSHOT (feeds regression testing)")
    print("=" * 70)

    for key, value in snapshot.items():
        shown = (
            f"{value:.6f}"
            if isinstance(value, float)
            else str(value)
        )
        print(f"  {key:<30} {shown}")

    print("=" * 70)


if __name__ == "__main__":
    main()
