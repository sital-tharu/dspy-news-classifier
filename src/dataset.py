import dspy
import random

def get_dataset(seed=42):
    politics = [
        dspy.Example(headline="Biden signs new climate bill into law", category="politics"),
        dspy.Example(headline="Senate passes infrastructure spending package", category="politics"),
        dspy.Example(headline="Prime Minister announces snap election", category="politics"),
        dspy.Example(headline="UN Security Council meets over conflict resolution", category="politics"),
        dspy.Example(headline="Congress debates new immigration reform bill", category="politics"),
        dspy.Example(headline="President vetoes military funding legislation", category="politics"),
        dspy.Example(headline="Governor declares state of emergency after protests", category="politics"),
    ]

    sports = [
        dspy.Example(headline="Lakers defeat Celtics in NBA Finals Game 7", category="sports"),
        dspy.Example(headline="Ronaldo scores hat trick in Champions League final", category="sports"),
        dspy.Example(headline="Serena Williams announces retirement from tennis", category="sports"),
        dspy.Example(headline="India wins Cricket World Cup in dramatic finish", category="sports"),
        dspy.Example(headline="Ferrari dominates Formula 1 race in Monaco", category="sports"),
        dspy.Example(headline="NFL quarterback signs record breaking contract", category="sports"),
        dspy.Example(headline="Olympics 2028 host city officially confirmed", category="sports"),
    ]

    tech = [
        dspy.Example(headline="Apple launches new AI chip for MacBook Pro", category="tech"),
        dspy.Example(headline="Google releases Gemini 2.0 with multimodal features", category="tech"),
        dspy.Example(headline="OpenAI raises 10 billion dollars in new funding round", category="tech"),
        dspy.Example(headline="Tesla autopilot software gets major safety update", category="tech"),
        dspy.Example(headline="Meta introduces new augmented reality glasses", category="tech"),
        dspy.Example(headline="Microsoft integrates AI copilot across all Office apps", category="tech"),
        dspy.Example(headline="Nvidia announces next generation GPU for data centers", category="tech"),
    ]

    devset_examples = [
        dspy.Example(headline="White House announces new foreign policy strategy", category="politics"),
        dspy.Example(headline="Parliament votes to raise minimum wage nationwide", category="politics"),
        dspy.Example(headline="Manchester United wins Premier League title", category="sports"),
        dspy.Example(headline="Swimmer breaks 100m freestyle world record", category="sports"),
        dspy.Example(headline="Samsung releases foldable phone with new display tech", category="tech"),
        dspy.Example(headline="SpaceX successfully launches 200th rocket mission", category="tech"),
    ]

    # Interleave categories so optimizer sees all 3 early
    trainset = []
    for p, s, t in zip(politics, sports, tech):
        trainset.append(p)
        trainset.append(s)
        trainset.append(t)

    # Apply consistent shuffle
    random.seed(seed)
    random.shuffle(trainset)

    trainset = [e.with_inputs("headline") for e in trainset]
    devset   = [e.with_inputs("headline") for e in devset_examples]

    return trainset, devset