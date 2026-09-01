from __future__ import annotations

from ecomas.agents import AgentSpec


MMLU_PRO_AGENTS = [
    AgentSpec(
        name="math_computation",
        role="Mathematics and computation",
        prompt="You specialize in mathematics, computer science, physics, and formal derivations. Prioritize definitions, formulas, boundary conditions, and calculation details.",
        example="If a question asks for the characteristic of a ring, start from the definition, check whether repeated addition sends every element to zero, and return 0 when no positive integer works.",
    ),
    AgentSpec(
        name="life_science",
        role="Life sciences",
        prompt="You specialize in medicine, biology, psychology, and health knowledge. Prioritize mechanisms, anatomy, physiology, pathology, and clinical conventions.",
        example="If a question asks about differences between catheters, first decide whether the difference is length, diameter, material, or use case, then map that finding to the answer options.",
    ),
    AgentSpec(
        name="social_institutions",
        role="Social institutions",
        prompt="You specialize in law, business, economics, politics, and institutional knowledge. Prioritize the governing body, institutional purpose, and constraints.",
        example="If a question asks about advertising regulation, identify standard prohibitions such as unsafe practices, distress or fear, and serious offence.",
    ),
    AgentSpec(
        name="humanities_behavior",
        role="Humanities and behavior",
        prompt="You specialize in history, philosophy, language, education, and human behavior. Prioritize conceptual relations, context, and common distractors.",
        example="If a question asks about a philosophical term, define the core concept first, then eliminate options that sound similar but belong to a different school or tradition.",
    ),
    AgentSpec(
        name="option_discriminator",
        role="Option discrimination",
        prompt="You are a multiple-choice verification specialist. Compare candidate options, look for qualifiers in the question, and revise earlier agents' answers when needed.",
        example="If an earlier answer only explains why one option seems plausible, eliminate the options one by one and output the single letter that best satisfies the question.",
    ),
]


MATH_500_AGENTS = [
    AgentSpec(
        name="algebra",
        role="Algebra",
        prompt="You specialize in equations, algebraic transformations, coordinate conversion, and symbolic calculation. Simplify the final answer into an exact expression whenever possible.",
        example="For polar coordinate conversion, compute r = sqrt(x^2 + y^2), determine theta from the quadrant, and check the angle range required by the problem.",
    ),
    AgentSpec(
        name="geometry",
        role="Geometry",
        prompt="You specialize in plane geometry, analytic geometry, trigonometry, and geometric relations. Prioritize constraints among objects and verify angles and lengths.",
        example="For circle and line problems, write the radius, tangent relation, similar triangles, or coordinate equations before solving the target quantity.",
    ),
    AgentSpec(
        name="number_theory",
        role="Number theory",
        prompt="You specialize in divisibility, congruences, prime factors, integer constructions, and Diophantine equations. Check integer constraints first.",
        example="For a linear Diophantine equation, find one solution first, then use the general solution to analyze gcd, ranges, or counts.",
    ),
    AgentSpec(
        name="combinatorics_probability",
        role="Combinatorics and probability",
        prompt="You specialize in counting, permutations, combinations, recurrences, probability, and expectation. Clarify the sample space and whether order matters.",
        example="For team partition problems, decide the team size and whether teams are distinguishable before using factorials or combinations to avoid overcounting.",
    ),
    AgentSpec(
        name="functions_inequalities",
        role="Functions and inequalities",
        prompt="You specialize in functions, sequences, extrema, inequalities, and estimates. Look for monotonicity, convexity, and equality conditions.",
        example="For optimization problems, convert the expression to a standard inequality or use derivatives, then verify whether equality can be attained.",
    ),
]


CHAOS_NLI_AGENTS = [
    AgentSpec(
        name="formal_semantics",
        role="Formal semantics",
        prompt="You specialize in truth conditions for NLI. Decide whether the premise necessarily supports, rules out, or leaves the hypothesis undetermined.",
        example="If the premise only says children are washing hands indoors and the hypothesis says they are at a ballgame, the hypothesis is not entailed and is usually neutral.",
    ),
    AgentSpec(
        name="lexical_syntax",
        role="Lexical syntax",
        prompt="You specialize in quantifiers, negation, coreference, tense, comparison structures, and lexical entailment. Search for fine-grained linguistic triggers.",
        example="Words such as not all, no, some, and never can change the entailment direction, so they must be checked token by token.",
    ),
    AgentSpec(
        name="commonsense_reasoning",
        role="Commonsense reasoning",
        prompt="You specialize in everyday knowledge, physical plausibility, social scripts, and event causality, but you must not overfill information missing from the premise.",
        example="If the premise says children are washing hands in a bathroom, commonsense cannot infer that they are at a ballgame; the missing location should remain unknown.",
    ),
    AgentSpec(
        name="pragmatic_explanation",
        role="Pragmatic explanation",
        prompt="You specialize in context, ellipsis, conversational implicature, and sources of human disagreement. Explain why an example may invite label uncertainty.",
        example="For spoken fragments or incomplete context, identify which missing information supports neutral instead of making an overconfident label choice.",
    ),
    AgentSpec(
        name="label_boundary",
        role="Label boundary judgment",
        prompt="You are an NLI label-boundary specialist. Make a strict final choice among entailment, neutral, and contradiction.",
        example="Entailment requires the hypothesis to be true in all reasonable worlds where the premise is true; contradiction requires the two to be impossible together; otherwise choose neutral.",
    ),
]


TASK_REGISTRY = {
    "mmlu_pro": {
        "display_name": "MMLU-Pro",
        "agents": MMLU_PRO_AGENTS,
        "steps": 4,
        "answer_format": "Return exactly one option letter A-J.",
    },
    "math500": {
        "display_name": "MATH-500",
        "agents": MATH_500_AGENTS,
        "steps": 6,
        "answer_format": "Return a concise exact mathematical answer, preferably LaTeX.",
    },
    "chaosnli": {
        "display_name": "ChaosNLI",
        "agents": CHAOS_NLI_AGENTS,
        "steps": 5,
        "answer_format": "Return exactly one label: entailment, neutral, or contradiction.",
    },
}
