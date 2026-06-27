import dspy
from dspy.teleprompt import BootstrapFewShot
from src.pipeline import NewsClassifierPipeline
from src.evaluation import classification_metric

def optimize_pipeline(trainset):
    """
    BootstrapFewShot:
    1. Runs pipeline on trainset examples
    2. Finds examples the model got RIGHT
    3. Injects them as few-shot examples into the prompt
    4. Returns an optimized pipeline
    """

    optimizer = BootstrapFewShot(
        metric=classification_metric,   # how to judge correctness
        max_bootstrapped_demos=4,       # max few-shot examples to add
        max_labeled_demos=4,            # max labeled examples to use
        max_rounds=1,                   # how many optimization rounds
    )

    print("🔄 Optimizing pipeline with BootstrapFewShot...")
    print("   This may take a few minutes...\n")

    optimized_pipeline = optimizer.compile(
        NewsClassifierPipeline(),       # fresh pipeline to optimize
        trainset=trainset               # learn from these examples
    )

    return optimized_pipeline