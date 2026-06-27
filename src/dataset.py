import dspy

def get_dataset():
    examples = [
        # --- POLITICS (7 examples) ---
        dspy.Example(headline="Biden signs new climate bill into law", category="politics"),
        dspy.Example(headline="Senate passes infrastructure spending package", category="politics"),
        dspy.Example(headline="Prime Minister announces snap election", category="politics"),
        dspy.Example(headline="UN Security Council meets over conflict resolution", category="politics"),
        dspy.Example(headline="Congress debates new immigration reform bill", category="politics"),
        dspy.Example(headline="President vetoes military funding legislation", category="politics"),
        dspy.Example(headline="Governor declares state of emergency after protests", category="politics"),

        # --- SPORTS (7 examples) ---
        dspy.Example(headline="Lakers defeat Celtics in NBA Finals Game 7", category="sports"),
        dspy.Example(headline="Ronaldo scores hat trick in Champions League final", category="sports"),
        dspy.Example(headline="Serena Williams announces retirement from tennis", category="sports"),
        dspy.Example(headline="India wins Cricket World Cup in dramatic finish", category="sports"),
        dspy.Example(headline="Ferrari dominates Formula 1 race in Monaco", category="sports"),
        dspy.Example(headline="NFL quarterback signs record breaking contract", category="sports"),
        dspy.Example(headline="Olympics 2028 host city officially confirmed", category="sports"),

        # --- TECH (6 examples) ---
        dspy.Example(headline="Apple launches new AI chip for MacBook Pro", category="tech"),
        dspy.Example(headline="Google releases Gemini 2.0 with multimodal features", category="tech"),
        dspy.Example(headline="OpenAI raises 10 billion dollars in new funding round", category="tech"),
        dspy.Example(headline="Tesla autopilot software gets major safety update", category="tech"),
        dspy.Example(headline="Meta introduces new augmented reality glasses", category="tech"),
        dspy.Example(headline="Microsoft integrates AI copilot across all Office apps", category="tech"),
    ]

    # Tell DSPy: "headline is the input, category is the label"
    examples = [e.with_inputs("headline") for e in examples]

    # Split: 14 for optimization, 6 for final evaluation
    trainset = examples[:14]
    devset   = examples[14:]

    return trainset, devset