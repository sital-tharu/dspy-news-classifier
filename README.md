# 📰 DSPy News Classifier

A news headline classifier built with [DSPy](https://dspy.ai) framework,
running locally on Ollama — no API keys required.

## What It Does
Classifies any news headline into:
- 🏛️ Politics
- ⚽ Sports  
- 💻 Tech

With a one-line explanation of *why*.

## Tech Stack
- **DSPy** — LLM programming framework
- **Ollama** — local LLM inference
- **llama3** — primary model used

## Project Structure
src/

├── signature.py   # DSPy Signature (input/output contract)

├── pipeline.py    # ChainOfThought classification module

├── dataset.py     # 27 labeled headlines (train + dev split)

├── evaluate.py    # Accuracy metric + evaluator

├── optimize.py    # BootstrapFewShot optimizer

├── experiment.py  # Multi-model comparison

└── save_load.py   # Save/load optimized pipeline

## Quickstart
```bash
# 1. Clone the repo
git clone https://github.com/sital-tharu/dspy-news-classifier
cd dspy-news-classifier

# 2. Install dependencies
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Start Ollama
ollama serve
ollama pull llama3

# 4. Run
python main.py
```

## Key Learnings

### What Worked
- DSPy Signatures cleanly separate task definition from prompting
- ChainOfThought improved reasoning transparency
- BootstrapFewShot auto-selects useful few-shot examples

### What Didn't Work (and Why)
- Optimization showed no gains on obvious headlines
- Small devset (6 examples) created a 100% ceiling
- phi3 got worse after optimization — few-shot demos added noise
- BootstrapFewShot picks examples in order, causing category bias

### Real Lessons
> DSPy optimization shines on HARD tasks with ambiguous examples.
> Data quality and balance matter more than optimizer choice.
> Always shuffle your trainset before optimization.

## Model Comparison Results

| Model      | Baseline | Optimized | Delta  |
|------------|----------|-----------|--------|
| llama3.2   | 100%     | 100%      | +0.0%  |
| phi3       | 100%     | 83.3%     | -16.7% |
| mistral    | 100%     | 100%      | +0.0%  |
| llama3     | 100%     | 100%      | +0.0%  |
| llama3.1   | 100%     | 100%      | +0.0%  |

## Next Steps
- [ ] RAG pipeline with real document retrieval
- [ ] Harder ambiguous headlines dataset
- [ ] MIPROv2 optimizer comparison
- [ ] Web UI with Gradio

## What is DSPy?
DSPy replaces hand-crafted prompts with **compiled programs**.
Instead of writing prompts, you define signatures and let the
optimizer find the best prompts automatically.