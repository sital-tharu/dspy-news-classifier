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
# After optimize_pipeline()
print("\n── Few-shot demos selected by optimizer ──")
try:
    # DSPy stores demos in the predictor
    predictor = optimized_pipeline.classify.predict
    demos = predictor.demos
    
    if not demos:
        print("No demos found — optimizer used labeled examples only")
    else:
        for i, demo in enumerate(demos):
            print(f"\nDemo {i+1}:")
            print(f"  Headline : {demo.headline}")
            print(f"  Category : {demo.category}")
            if hasattr(demo, 'reasoning'):
                print(f"  Reasoning: {demo.reasoning[:80]}...")
except Exception as e:
    # Fallback — inspect the full pipeline structure
    print("\nInspecting pipeline structure...")
    print(optimized_pipeline.classify)
    print("\nNamed predictors:")
    for name, predictor in optimized_pipeline.named_predictors():
        print(f"  {name}: {predictor}")
        if hasattr(predictor, 'demos'):
            for i, demo in enumerate(predictor.demos):
                print(f"\n  Demo {i+1}:")
                print(f"    Headline : {demo.get('headline', 'N/A')}")
                print(f"    Category : {demo.get('category', 'N/A')}")