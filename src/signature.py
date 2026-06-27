# src/signature.py
import dspy

class NewsClassifier(dspy.Signature):
    """Classify a news headline into a category and explain why."""
    
    headline: str = dspy.InputField(desc="A news headline to classify")
    category: str = dspy.OutputField(desc="One of: politics, sports, tech")
    reason: str = dspy.OutputField(desc="One-line explanation for the classification")
