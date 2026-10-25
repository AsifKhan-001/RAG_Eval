
"""
Operational eval: COST.

Measures real input/output token usage from the Gemini model,
calculates estimated API cost, and projects daily/monthly expenses.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import prompt, llm

load_dotenv()

# Stop before StrOutputParser() so token usage metadata survives.
measured_chain = prompt | llm

# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 3

# Gemini pricing: USD per 1M tokens.
# Update these values using the current pricing for your exact model.
# These are illustrative paid-tier estimates, NOT a guarantee of current pricing.
PRICE_INPUT_PER_1M = 0.30
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 2.50

# Business projection knobs
QUERIES_PER_DAY = 2000
USD_TO_INR = 95.0

# Estimated cost budget per query
COST_BUDGET_PER_QUERY_USD = 0.0015

# ============================================================
# 3. TOKEN MEASUREMENT
# ============================================================
def measure_tokens(pipeline, question):
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

    # Cache field names can vary by integration/version.
    cached_tokens = (
        details.get("cache_read")
        or details.get("cache_read_input_tokens")
        or 0
    )

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }


# ============================================================
# 4. COST MATH
# ============================================================
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


# ============================================================
# 5. BENCHMARK LOOP
# ============================================================
def benchmark(pipeline):
    rows = []

    print("Measuring Gemini token usage...")

    for question in QUESTIONS:
        for _ in range(REPEATS):
            tok = measure_tokens(pipeline, question)

            cost = cost_usd(
                tok["input"],
                tok["output"],
                tok["cached"],
            )

            rows.append({
                **tok,
                **{
                    f"cost_{key}": value
                    for key, value in cost.items()
                },
            })

    return rows


# ============================================================
# 6. AGGREGATE + REPORT
# ============================================================
def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def report(rows):
    if not rows:
        print("No measurements available.")
        return

    n = len(rows)

    avg_in = avg(rows, "input")
    avg_out = avg(rows, "output")
    avg_cached = avg(rows, "cached")
    avg_cost = avg(rows, "cost_total")

    min_cost = min(r["cost_total"] for r in rows)
    max_cost = max(r["cost_total"] for r in rows)

    avg_cost_in = (
        avg(rows, "cost_input")
        + avg(rows, "cost_cached")
    )
    avg_cost_out = avg(rows, "cost_output")

    out_share = (
        100 * avg_cost_out / avg_cost
        if avg_cost
        else 0
    )

    print("\n" + "=" * 70)
    print("COST (Google Gemini)")
    print("=" * 70)
    print(f"samples                : {n}")
    print(
        f"avg input tokens       : {avg_in:8.0f} "
        f"({avg_cached:.0f} cached)"
    )
    print(f"avg output tokens      : {avg_out:8.0f}")
    print("-" * 70)

    print(
        f"estimated cost/query   : ${avg_cost:.6f} "
        f"(Rs {avg_cost * USD_TO_INR:.4f})"
    )
    print(
        f"   min / max           : ${min_cost:.6f} "
        f"/ ${max_cost:.6f}"
    )
    print(
        f"   input vs output     : "
        f"{100 - out_share:.0f}% input / "
        f"{out_share:.0f}% output"
    )

    print("-" * 70)

    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30

    print(f"projection @ {QUERIES_PER_DAY}/day:")
    print(
        f"   per day             : ${daily:8.2f} "
        f"(Rs {daily * USD_TO_INR:8.2f})"
    )
    print(
        f"   per month           : ${monthly:8.2f} "
        f"(Rs {monthly * USD_TO_INR:8.2f})"
    )

    print("=" * 70)

    verdict = (
        "PASS"
        if avg_cost <= COST_BUDGET_PER_QUERY_USD
        else "FAIL"
    )

    print(
        f"BUDGET: cost/query <= "
        f"${COST_BUDGET_PER_QUERY_USD:.6f} "
        f"-> ${avg_cost:.6f} [{verdict}]"
    )
    print("=" * 70)

    print(
        "Note: This estimate uses configured paid-tier rates. "
        "Free-tier requests within quota may incur no API charge. "
        "Check current Gemini pricing and usage metadata before "
        "using this estimate for billing decisions."
    )


# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():
    pipeline = RagPipeline()
    rows = benchmark(pipeline)
    report(rows)


if __name__ == "__main__":
    main()
