import dspy
from src.signature import NewsClassifier

class NewsClassifierPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(NewsClassifier)

    def forward(self, headline):
        result = self.classify(headline=headline)
        return result