from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.gemini_judge import GeminiJudge

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "gemini-2.5-flash"
THRESHOLD = 0.7


def run(retriever):
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    judge = GeminiJudge(JUDGE_MODEL)
    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": "gemini-embedding-001",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    return run(RerankingRetriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())