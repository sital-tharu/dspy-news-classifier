import dspy
from src.pipeline import NewsClassifierPipeline
from src.dataset import get_dataset
from src.evaluation import evaluate_pipeline
from src.optimize import optimize_pipeline

# Configure Ollama
lm = dspy.LM("ollama/llama3", api_base="http://localhost:11434")
dspy.configure(lm=lm)

# # Run a quick test
# pipeline = NewsClassifierPipeline()
# result = pipeline(headline="Apple launches new AI chip for MacBook Pro")

# print(f"Category : {result.category}")
# print(f"Reason   : {result.reason}")

# trainset, devset = get_dataset()
# print(f"Trainset size : {len(trainset)}")
# print(f"Devset size   : {len(devset)}")
# print(f"\nSample example:")
# print(f"  Headline : {trainset[0].headline}")
# print(f"  Category : {trainset[0].category}")

# ── Build Pipeline ─────────────────────────────────────────────────
# pipeline = NewsClassifierPipeline()

# # ── Quick single test ──────────────────────────────────────────────
# print("── Single Prediction Test ──")
# result = pipeline(headline="Apple launches new AI chip for MacBook Pro")
# print(f"Category : {result.category}")
# print(f"Reason   : {result.reason}\n")

# # ── Evaluate on devset ─────────────────────────────────────────────
# print("── Baseline Evaluation ──")
# baseline_score = evaluate_pipeline(pipeline, devset)
# print(f"\n✅ Baseline Accuracy: {baseline_score}")


# ── Load Data ──────────────────────────────────────────────────────
trainset, devset = get_dataset()
print(f"Trainset : {len(trainset)} examples")
print(f"Devset   : {len(devset)} examples\n")

# ── Baseline ───────────────────────────────────────────────────────
print("── Baseline Evaluation ──")
baseline_pipeline = NewsClassifierPipeline()
baseline_score = evaluate_pipeline(baseline_pipeline, devset)
print(f"✅ Baseline Accuracy: {baseline_score.score:.1f}%\n")

# ── Optimize ───────────────────────────────────────────────────────
optimized_pipeline = optimize_pipeline(trainset)

# ── Evaluate Optimized ─────────────────────────────────────────────
print("\n── Optimized Evaluation ──")
optimized_score = evaluate_pipeline(optimized_pipeline, devset)
print(f"✅ Optimized Accuracy: {optimized_score.score:.1f}%")

# ── Compare ────────────────────────────────────────────────────────
print("\n── Results Comparison ──")
print(f"  Before optimization : {baseline_score.score:.1f}%")
print(f"  After optimization  : {optimized_score.score:.1f}%")
improvement = optimized_score.score - baseline_score.score
print(f"  Improvement         : +{improvement:.1f}%")