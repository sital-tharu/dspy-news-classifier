import dspy
from dspy.evaluate import Evaluate
from src.pipeline import NewsClassifierPipeline
from src.dataset import get_dataset

# ── 1. The Metric Function ──────────────────────────────────────────
def classification_metric(example, prediction, trace=None):
    """
    Returns True if predicted category matches expected category.
    - example    → the labeled data (ground truth)
    - prediction → what the pipeline returned
    - trace      → used internally by DSPy optimizer (ignore for now)
    """
    expected  = example.category.strip().lower()
    predicted = prediction.category.strip().lower()

    is_correct = expected == predicted

    return is_correct


# ── 2. The Evaluator ───────────────────────────────────────────────
def evaluate_pipeline(pipeline, devset):
    """
    Runs the pipeline on every example in devset
    and prints the accuracy score.
    """
    evaluator = Evaluate(
        devset=devset,
        metric=classification_metric,
        num_threads=1,          # 1 = sequential (safe for Ollama)
        display_progress=True,  # shows a progress bar
        display_table=True,     # shows per-example results
    )

    score = evaluator(pipeline)
    return score