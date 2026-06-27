import dspy
from src.pipeline import NewsClassifierPipeline
from src.dataset import get_dataset

# Configure Ollama
lm = dspy.LM("ollama/llama3.2", api_base="http://localhost:11434")
dspy.configure(lm=lm)

# # Run a quick test
# pipeline = NewsClassifierPipeline()
# result = pipeline(headline="Apple launches new AI chip for MacBook Pro")

# print(f"Category : {result.category}")
# print(f"Reason   : {result.reason}")

trainset, devset = get_dataset()
print(f"Trainset size : {len(trainset)}")
print(f"Devset size   : {len(devset)}")
print(f"\nSample example:")
print(f"  Headline : {trainset[0].headline}")
print(f"  Category : {trainset[0].category}")