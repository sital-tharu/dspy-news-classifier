import dspy
import os

def save_pipeline(pipeline, path="models/optimized_pipeline.json"):
    os.makedirs("models", exist_ok=True)
    pipeline.save(path)
    print(f"✅ Pipeline saved to {path}")

def load_pipeline(pipeline_class, path="models/optimized_pipeline.json"):
    pipeline = pipeline_class()
    pipeline.load(path)
    print(f"✅ Pipeline loaded from {path}")
    return pipeline