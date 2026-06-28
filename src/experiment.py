# src/experiment.py
import dspy
from src.pipeline import NewsClassifierPipeline
from src.dataset import get_dataset
from src.evaluation import evaluate_pipeline
from src.optimize import optimize_pipeline

MODELS = [
    "ollama/llama3.2",
    "ollama/phi3",
    "ollama/mistral",
    "ollama/llama3",
    "ollama/llama3.1:8b",
]

def run_experiment():
    trainset, devset = get_dataset()
    results = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"🧪 Testing model: {model_name}")
        print(f"{'='*60}")

        try:
            # Configure model
            lm = dspy.LM(model_name, api_base="http://localhost:11434")
            dspy.configure(lm=lm)

            # Baseline
            print(f"\n── Baseline ──")
            baseline_pipeline = NewsClassifierPipeline()
            baseline_score = evaluate_pipeline(baseline_pipeline, devset)

            # Optimize
            print(f"\n── Optimizing ──")
            optimized_pipeline = optimize_pipeline(trainset)

            # Optimized score
            print(f"\n── Optimized ──")
            optimized_score = evaluate_pipeline(optimized_pipeline, devset)

            results.append({
                "model"    : model_name.replace("ollama/", ""),
                "baseline" : round(baseline_score.score, 1),
                "optimized": round(optimized_score.score, 1),
                "delta"    : round(optimized_score.score - baseline_score.score, 1),
                "status"   : "✅"
            })

        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            results.append({
                "model"    : model_name.replace("ollama/", ""),
                "baseline" : 0,
                "optimized": 0,
                "delta"    : 0,
                "status"   : "❌ Failed"
            })

    return results


def print_results_table(results):
    print(f"\n{'='*60}")
    print(f"{'📊 FINAL COMPARISON TABLE':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Baseline':>10} {'Optimized':>10} {'Delta':>8} {'Status':>8}")
    print(f"{'-'*60}")
    for r in results:
        delta_str = f"+{r['delta']}%" if r['delta'] >= 0 else f"{r['delta']}%"
        print(f"{r['model']:<20} {str(r['baseline'])+'%':>10} {str(r['optimized'])+'%':>10} {delta_str:>8} {r['status']:>8}")
    print(f"{'='*60}")

    # Best model
    successful = [r for r in results if r['status'] == '✅']
    if successful:
        best_baseline  = max(successful, key=lambda x: x['baseline'])
        best_optimized = max(successful, key=lambda x: x['optimized'])
        best_improver  = max(successful, key=lambda x: x['delta'])

        print(f"\n🏆 Best baseline accuracy : {best_baseline['model']} ({best_baseline['baseline']}%)")
        print(f"🏆 Best optimized accuracy: {best_optimized['model']} ({best_optimized['optimized']}%)")
        print(f"🏆 Most improved by DSPy  : {best_improver['model']} (+{best_improver['delta']}%)")