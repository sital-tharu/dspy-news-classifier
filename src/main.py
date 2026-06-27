import dspy
from src.pipeline import NewsClassifierPipeline

# Configure Ollama
lm = dspy.LM("ollama/llama3.2", api_base="http://localhost:11434")
dspy.configure(lm=lm)

# Run a quick test
pipeline = NewsClassifierPipeline()
result = pipeline(headline="Apple launches new AI chip for MacBook Pro")

print(f"Category : {result.category}")
print(f"Reason   : {result.reason}")