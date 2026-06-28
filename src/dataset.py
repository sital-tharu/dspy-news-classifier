import dspy

def get_dataset():
    examples = [
        # --- POLITICS (8 examples) ---
        dspy.Example(headline="Biden signs new climate bill into law", category="politics"),
        dspy.Example(headline="Senate passes infrastructure spending package", category="politics"),
        dspy.Example(headline="Prime Minister announces snap election", category="politics"),
        dspy.Example(headline="UN Security Council meets over conflict resolution", category="politics"),
        dspy.Example(headline="Congress debates new immigration reform bill", category="politics"),
        dspy.Example(headline="President vetoes military funding legislation", category="politics"),
        dspy.Example(headline="Governor declares state of emergency after protests", category="politics"),
        dspy.Example(headline="Elon Musk tweets about government AI regulation", category="politics"),

        # --- SPORTS (8 examples) ---
        dspy.Example(headline="Lakers defeat Celtics in NBA Finals Game 7", category="sports"),
        dspy.Example(headline="Ronaldo scores hat trick in Champions League final", category="sports"),
        dspy.Example(headline="Serena Williams announces retirement from tennis", category="sports"),
        dspy.Example(headline="India wins Cricket World Cup in dramatic finish", category="sports"),
        dspy.Example(headline="Ferrari dominates Formula 1 race in Monaco", category="sports"),
        dspy.Example(headline="NFL quarterback signs record breaking contract", category="sports"),
        dspy.Example(headline="Olympics 2028 host city officially confirmed", category="sports"),
        dspy.Example(headline="Olympic athletes use AI training tools to break records", category="sports"),

        # --- TECH (8 examples) ---
        dspy.Example(headline="Apple launches new AI chip for MacBook Pro", category="tech"),
        dspy.Example(headline="Google releases Gemini 2.0 with multimodal features", category="tech"),
        dspy.Example(headline="OpenAI raises 10 billion dollars in new funding round", category="tech"),
        dspy.Example(headline="Tesla autopilot software gets major safety update", category="tech"),
        dspy.Example(headline="Meta introduces new augmented reality glasses", category="tech"),
        dspy.Example(headline="Microsoft integrates AI copilot across all Office apps", category="tech"),
        dspy.Example(headline="Tesla CEO appears before Senate committee on autopilot", category="politics"),
        dspy.Example(headline="Nvidia announces next generation GPU for data centers", category="tech"),
    ]

    examples = [e.with_inputs("headline") for e in examples]

    # 21 total → 18 train, 6 dev (2 per category)
    trainset = examples[:18]
    devset   = examples[18:]

    return trainset, devset
