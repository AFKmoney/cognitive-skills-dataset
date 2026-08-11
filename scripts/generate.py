#!/usr/bin/env python3
"""
Cognitive Skills Dataset Generator v2 — fast & scalable
========================================================
Generates a massive JSONL training dataset for LLM fine-tuning across 5 cognitive skills:
  1. THINKING        (critical/analytical/systems/lateral/strategic thinking)
  2. REASONING       (deductive/inductive/abductive/analogical/causal/probabilistic)
  3. SPEAKING        (clear expression/rhetoric/persuasion/technical comm)
  4. UNDERSTANDING   (comprehension/synthesis/context/perspective-taking)
  5. CODING          (algorithmic thinking/code review/debugging/system design)

Layout: 5 skills x 20 subtopics = 100 JSONL files.
Each file contains ~2500 chat-formatted entries.
Total target: ~250,000 entries, ~250M+ tokens.

All content is in English and instruction-tuning oriented (chat messages format).

Architecture for variety:
  - 15 prompt templates per skill
  - 10 audiences x 10 perspectives x 10 formats = 1000 variation tuples
  - Per subtopic, each entry gets a unique (template_idx, audience, perspective, format) tuple
  - Dedup by user-content hash
  - Long, structured assistant responses (~400-700 tokens each)
"""

import json
import os
import sys
import random
import hashlib
import time
from multiprocessing import Pool, cpu_count

random.seed(42)
OUT_DIR = "/home/z/my-project/cognitive-skills/data"
os.makedirs(OUT_DIR, exist_ok=True)

SKILLS = {
    "thinking": [
        "critical_thinking", "analytical_thinking", "systems_thinking",
        "lateral_thinking", "strategic_thinking", "design_thinking",
        "first_principles", "decision_making", "problem_framing",
        "mental_models", "cognitive_biases", "hypothesis_testing",
        "evidence_evaluation", "counterfactual_thinking", "metacognition",
        "root_cause_analysis", "tradeoff_analysis", "abstraction",
        "synthesis", "creative_thinking",
    ],
    "reasoning": [
        "deductive_reasoning", "inductive_reasoning", "abductive_reasoning",
        "analogical_reasoning", "causal_reasoning", "probabilistic_reasoning",
        "logical_fallacies", "syllogisms", "boolean_logic",
        "modal_logic", "statistical_reasoning", "bayesian_reasoning",
        "counterfactual_reasoning", "diagnostic_reasoning", "spatial_reasoning",
        "temporal_reasoning", "moral_reasoning", "ethical_dilemmas",
        "counterargument_construction", "proof_verification",
    ],
    "speaking": [
        "clear_expression", "structured_communication", "persuasion",
        "rhetoric", "storytelling", "technical_explanation",
        "executive_communication", "negotiation", "conflict_resolution",
        "active_listening", "feedback_delivery", "public_speaking",
        "interview_skills", "explanatory_teaching", "concise_writing",
        "diplomacy", "question_formulation", "analogy_crafting",
        "audience_adaptation", "tone_calibration",
    ],
    "understanding": [
        "reading_comprehension", "literal_interpretation", "inferential_comprehension",
        "contextual_understanding", "perspective_taking", "empathy",
        "metaphor_interpretation", "ambiguity_resolution", "implicit_meaning",
        "cultural_context", "discourse_analysis", "summarization",
        "synthesis_across_sources", "fact_checking", "source_evaluation",
        "intention_recognition", "subtext_detection", "pragmatic_understanding",
        "concept_extraction", "semantic_relationships",
    ],
    "coding": [
        "algorithmic_thinking", "data_structures", "complexity_analysis",
        "code_review", "debugging", "system_design",
        "api_design", "design_patterns", "refactoring",
        "testing_strategies", "concurrency", "error_handling",
        "code_readability", "documentation", "version_control",
        "performance_optimization", "security_practices", "database_design",
        "functional_programming", "object_oriented_design",
    ],
}

AUDIENCES = [
    "a beginner who has never seen the concept before",
    "an intermediate practitioner looking to deepen understanding",
    "an advanced expert who wants a rigorous treatment",
    "a student preparing for an exam",
    "a professional applying the concept at work",
    "a teacher preparing a lesson plan",
    "a self-learner studying independently",
    "a manager who needs a high-level overview",
    "a researcher investigating the underlying theory",
    "a reviewer evaluating the quality of the reasoning",
]

PERSPECTIVES = [
    "from a theoretical foundations angle",
    "from a practical application angle",
    "from a historical development angle",
    "from a comparative perspective (contrasting alternatives)",
    "from a critical/limitation-focused perspective",
    "from a teach-the-skill angle",
    "from an everyday-life angle",
    "from a professional/workplace angle",
    "from a debugging/error-correction angle",
    "from an optimization/mastery angle",
]

FORMATS = [
    "step-by-step procedure",
    "bulleted checklist",
    "worked example with concrete numbers",
    "scenario walkthrough",
    "Q&A dialogue",
    "compare-and-contrast table description",
    "decision tree narrative",
    "common mistakes and corrections",
    "before/after transformation",
    "principles-then-application",
]

def _msg(role, content):
    return {"role": role, "content": content}

def _wrap(user, assistant):
    return {"messages": [_msg("user", user), _msg("assistant", assistant)]}

# ----------------------------------------------------------------------------
# SKILL-SPECIFIC DEFINITIONS (for rich, varied content per subtopic)
# ----------------------------------------------------------------------------

THINKING_DEFS = {
    "critical_thinking": {
        "def": "the disciplined practice of evaluating claims, arguments, and evidence by examining their assumptions, logic, and supporting data before accepting them as true or acting on them",
        "principles": ["Suspend judgment until evidence is in", "Distinguish claims from evidence", "Question the source and its incentives", "Seek disconfirming evidence actively", "Calibrate confidence to evidence strength"],
        "examples": ["Evaluating whether a startup pitch's market-size claim is justified", "Assessing a clinical trial's conclusion before adopting the treatment", "Deciding whether a customer's complaint reflects a real product flaw"],
        "mistakes": ["Anchoring on the first plausible explanation", "Confirmation bias — seeking only supporting evidence", "Confusing correlation with causation", "Failing to consider source incentives", "Treating absence of disproof as proof"],
    },
    "analytical_thinking": {
        "def": "the practice of breaking a complex situation into its constituent parts, examining each part and its relationships to the others, and reconstructing the whole with explicit understanding of how it works",
        "principles": ["Decompose before diagnosing", "Measure each component separately", "Quantify, don't qualify", "Look for outliers and explain them", "Test hypotheses with controlled experiments"],
        "examples": ["Diagnosing a revenue drop by decomposing into traffic, conversion, AOV", "Analyzing why a query is slow by decomposing into parse, plan, execute, fetch", "Understanding churn by decomposing by cohort, segment, and reason"],
        "mistakes": ["Aggregating before decomposing", "Stopping at the first abnormal number", "Confusing segment shifts with real changes (Simpson's paradox)", "Ignoring time-series structure", "Cherry-picking the favorable segment"],
    },
    "systems_thinking": {
        "def": "the practice of understanding a situation as a system of interacting parts with feedback loops, delays, and emergent behavior, rather than as a collection of independent events",
        "principles": ["Map the feedback loops, not just the parts", "Identify delays — they cause oscillation", "Find leverage points, not symptoms", "Watch for unintended consequences", "Distinguish balancing and reinforcing loops"],
        "examples": ["Why widening roads increases congestion (induced demand)", "Why code review backlogs grow despite faster reviewers", "Why adding more developers to a late project makes it later (Brooks' Law)"],
        "mistakes": ["Treating symptoms as causes", "Optimizing a part at the expense of the whole", "Ignoring feedback loops", "Assuming linear cause and effect", "Forgetting delays between action and effect"],
    },
    "lateral_thinking": {
        "def": "the practice of solving problems through indirect, creative, or non-obvious approaches, rather than through step-by-step logical progression",
        "principles": ["Challenge assumptions explicitly", "Generate many alternatives before evaluating", "Use random entry to break patterns", "Reverse the problem", "Borrow from distant domains"],
        "examples": ["Solving a library's teen-engagement problem with zero budget by trading volunteer hours for fine forgiveness", "Reducing customer-support load by writing the FAQ from the customer's questions, not the company's categories", "Designing a museum exhibit by starting from the visitor's emotion, not the artifact"],
        "mistakes": ["Evaluating ideas during generation", "Stopping at the first acceptable idea", "Confusing wild with creative", "Forgetting that lateral ideas still need to work", "Dismissing constraints as 'creative blocks'"],
    },
    "strategic_thinking": {
        "def": "the practice of making decisions whose value depends on the actions, reactions, and resources of other actors, over a long time horizon, under uncertainty",
        "principles": ["Pick battles you can win; avoid battles you must win", "Compete on a dimension the opponent can't match", "Think in tempo, not just in moves", "Make reversible decisions fast; irreversible ones slowly", "Plan the endgame before the opening"],
        "examples": ["A startup choosing a beachhead segment the incumbent ignores", "A coffee shop owner deciding whether to open a second location", "A negotiator timing an offer to maximize leverage"],
        "mistakes": ["Confusing strategy with planning", "Optimizing for the current move, not the sequence", "Copying the leader's strategy", "Ignoring the competitor's incentives", "Treating reversible decisions as irreversible and vice versa"],
    },
    "design_thinking": {
        "def": "a human-centered approach to problem-solving that emphasizes empathy with users, iterative prototyping, and tolerance for ambiguity in the search for novel solutions",
        "principles": ["Empathize before ideating", "Frame the problem as a question", "Prototype to think, not to demonstrate", "Test with real users, not stakeholders", "Iterate quickly; fail cheaply"],
        "examples": ["Redesigning a hospital intake form by shadowing patients, not interviewing administrators", "Solving 'people don't read release notes' by prototyping a one-line summary format", "Improving onboarding by watching first-time users, not surveying them"],
        "mistakes": ["Skipping empathy research", "Treating prototypes as demos", "Testing with friendly users instead of real ones", "Falling in love with the first idea", "Confusing 'design' with 'visual design'"],
    },
    "first_principles": {
        "def": "the practice of reasoning from fundamental truths that cannot be further decomposed, rather than by analogy to existing solutions",
        "principles": ["Identify the absolute constraints (physics, math, definitions)", "Strip away every assumption that isn't a constraint", "Reconstruct the solution from the constraints", "Question analogies — they hide assumptions", "Distinguish 'this is how it's done' from 'this is how it must be'"],
        "examples": ["Musk's battery-cost reasoning: a battery's material components cost $80; therefore a $600/kWh battery is not a physics limit but a manufacturing one", "Reconsidering 'a restaurant needs a kitchen' to enable cloud kitchens", "Reconsidering 'a university needs a campus' to enable online degrees"],
        "mistakes": ["Stopping at analogy", "Mistaking convention for principle", "Ignoring real constraints in the name of 'thinking fresh'", "Treating first principles as a license to ignore domain expertise", "Confusing first principles with first impressions"],
    },
    "decision_making": {
        "def": "the practice of choosing among alternatives under uncertainty, with explicit attention to the quality of the process (not just the outcome)",
        "principles": ["Distinguish reversible from irreversible decisions", "Use a decision journal to capture reasoning at decision time", "Pre-mortem before committing", "Calibrate confidence to evidence", "Decide who decides — and on what basis"],
        "examples": ["Choosing between two job offers using a weighted scoring rubric", "Deciding whether to ship a feature with incomplete test coverage", "Choosing a cloud provider with a 3-year horizon"],
        "mistakes": ["Outcome bias — judging the decision by the outcome", "Sunk-cost fallacy", "Analysis paralysis on reversible decisions", "Premature commitment on irreversible ones", "Deciding by consensus when speed matters more than buy-in"],
    },
    "problem_framing": {
        "def": "the practice of stating a problem in a form that makes the solution space productive — neither so narrow that good options are excluded, nor so broad that the problem becomes intractable",
        "principles": ["Reframe the problem at least three ways before solving", "Identify whose problem this is", "Distinguish the symptom from the problem", "Make the framing's assumptions explicit", "Choose the framing whose answer would change your action"],
        "examples": ["Reframing 'users don't click the button' as 'users don't understand the value proposition behind the button'", "Reframing 'we need more engineers' as 'we need higher throughput per engineer'", "Reframing 'the model is hallucinating' as 'the model is being asked questions outside its training distribution'"],
        "mistakes": ["Accepting the first framing", "Framing the problem as the symptom", "Framing to fit a preferred solution", "Framing too narrowly to exclude creative options", "Framing too broadly to be actionable"],
    },
    "mental_models": {
        "def": "the practice of using compressed representations of how the world works (models) to reason about situations, while remaining aware that every model is wrong in some respect",
        "principles": ["Carry many models; use the one that fits", "Name the model in use to make it auditable", "Update the model when reality contradicts it", "Know each model's failure modes", "Avoid over-applying a single model"],
        "examples": ["Using 'opportunity cost' to evaluate a 'free' internal tool", "Using 'regression to the mean' to avoid overreacting to extreme performance quarters", "Using 'second-order effects' to predict why a 'fix' causes a new problem"],
        "mistakes": ["Single-model thinking", "Applying a model outside its domain of validity", "Treating the model as the territory", "Forgetting to update on disconfirming evidence", "Choosing the model after the conclusion (motivated reasoning)"],
    },
    "cognitive_biases": {
        "def": "systematic patterns of deviation from rationality in judgment, often unconscious, that affect decisions in predictable ways",
        "principles": ["Name the bias when you suspect it", "Design decision processes that mitigate common biases", "Use checklists, not willpower, to overcome bias", "Be especially vigilant under time pressure", "Calibrate confidence down when bias risk is high"],
        "examples": ["Anchoring on the first number in a negotiation", "Availability bias overestimating recent events", "Survivorship bias ignoring failed projects when studying success"],
        "mistakes": ["Using 'bias' as a thought-terminating cliché", "Assuming you can debias yourself by awareness alone", "Confusing bias with preference", "Forgetting that bias cuts both ways", "Using bias accusations as a rhetorical weapon"],
    },
    "hypothesis_testing": {
        "def": "the practice of formulating a falsifiable hypothesis, designing a test that could disprove it, and updating one's belief based on the result",
        "principles": ["Make the hypothesis falsifiable", "Design the test before collecting data", "Pre-register the analysis to avoid p-hacking", "Specify what would falsify, not just confirm", "Update incrementally with Bayes-like reasoning"],
        "examples": ["Testing whether a UI change increases conversion (A/B test)", "Testing whether a memory leak is caused by a specific code path", "Testing whether a hiring rubric predicts on-the-job performance"],
        "mistakes": ["Vague hypotheses that can't be falsified", "Testing after collecting data (HARKing)", "Multiple comparisons without correction", "Confusing statistical significance with practical significance", "Failing to specify falsification criteria upfront"],
    },
    "evidence_evaluation": {
        "def": "the practice of assessing the strength, relevance, and credibility of evidence before letting it change one's beliefs",
        "principles": ["Rate evidence by source, design, and replication", "Distinguish anecdote from data", "Weight evidence by its independence", "Be skeptical of evidence that confirms your priors", "Track the provenance of every claim"],
        "examples": ["Evaluating a single dramatic case study vs a meta-analysis", "Evaluating a vendor-sponsored white paper", "Evaluating a colleague's 'gut feeling' against dashboard data"],
        "mistakes": ["Treating all evidence as equal", "Discounting unfamiliar evidence sources", "Overweighting vivid evidence", "Confusing peer review with correctness", "Forgetting to check replication"],
    },
    "counterfactual_thinking": {
        "def": "the practice of reasoning about what would have happened under alternatives to the actual course of events, to evaluate causality and learn from outcomes",
        "principles": ["Construct realistic counterfactuals, not fanciful ones", "Use 'but-for' tests for causality", "Compare to the most plausible alternative, not the ideal one", "Watch for hindsight bias when constructing counterfactuals", "Use pre-mortems to generate them prospectively"],
        "examples": ["'If we hadn't shipped the feature, would the churn have been lower?'", "'If we had hired the other candidate, would the project have shipped on time?'", "'If the bug had been caught in review, would the outage still have happened?'"],
        "mistakes": ["Hindsight bias — assuming the alternative was knowable", "Constructing self-serving counterfactuals", "Comparing to ideal rather than plausible alternative", "Confusing counterfactual with causal claim", "Forgetting that counterfactuals are unobservable"],
    },
    "metacognition": {
        "def": "thinking about one's own thinking — monitoring, evaluating, and regulating one's cognitive processes while engaged in them",
        "principles": ["Notice when you're confused", "Track your confidence explicitly", "Review your reasoning for known bias patterns", "Distinguish 'I don't know' from 'I haven't thought about it'", "Pause to ask 'why am I so sure?'"],
        "examples": ["Catching yourself rationalizing a decision after the fact", "Noticing your attention slipping during a complex read", "Detecting that you've stopped reading carefully and started skimming"],
        "mistakes": ["Confusing fluency with understanding", "Overestimating retention without testing", "Treating metacognition as navel-gazing", "Forgetting to act on metacognitive signals", "Using metacognition to delay action indefinitely"],
    },
    "root_cause_analysis": {
        "def": "the practice of tracing an observed problem back through its causal chain to the underlying cause whose removal would prevent recurrence",
        "principles": ["Use '5 Whys' to push past the surface", "Distinguish proximate cause from root cause", "Look for systemic causes, not individual ones", "Verify the root cause would have produced the symptom", "Address the root cause, not the symptom"],
        "examples": ["'Server crashed' → 'OOM' → 'memory leak' → 'unclosed cursor' → 'no lint rule for cursor cleanup'", "'Customer churned' → 'support took 3 days' → 'ticket was misrouted' → 'no triage SLA'", "'Bug escaped to prod' → 'no test for this path' → 'test plan didn't list this case' → 'no test plan template'"],
        "mistakes": ["Stopping at the first plausible cause", "Blaming individuals instead of systems", "Skipping verification of the root cause", "Treating '5 Whys' as a literal count rather than a discipline", "Addressing the root cause but leaving the symptom-remediation undone"],
    },
    "tradeoff_analysis": {
        "def": "the practice of explicitly naming the costs of each alternative and deciding which cost to accept, rather than searching for a free option that doesn't exist",
        "principles": ["Name the tradeoff, don't hide it", "Quantify where possible", "Identify which constraint is binding", "Make the tradeoff explicit to stakeholders", "Revisit tradeoffs as constraints change"],
        "examples": ["Latency vs cost (cache more, pay more)", "Time-to-market vs quality (ship now, fix later — sometimes)", "Consistency vs availability (CAP theorem)", "Specialization vs flexibility"],
        "mistakes": ["Pretending there's no tradeoff", "Choosing by default rather than by analysis", "Confusing reversible with irreversible tradeoffs", "Forgetting to revisit when constraints change", "Optimizing one dimension past diminishing returns"],
    },
    "abstraction": {
        "def": "the practice of identifying the essential structure of a problem while suppressing incidental details, so the solution applies to a class of problems rather than a single instance",
        "principles": ["Abstract only after 2-3 concrete instances", "Name abstractions deliberately — names outlive code", "Push abstractions to the layer where they belong", "Resist premature abstraction", "Refactor abstractions when the underlying pattern changes"],
        "examples": ["Extracting a 'fetch with retry' helper after copy-pasting it three times", "Recognizing that three different endpoints share a 'paginated response' abstraction", "Abstracting 'logging' so the same code works for stdout, file, and network"],
        "mistakes": ["Premature abstraction (abstracting before the pattern is clear)", "Wrong-layer abstraction (in the wrong module)", "Over-abstraction (the abstraction is more complex than the duplication)", "Under-abstraction (copy-paste because 'I'll abstract later')", "Abstraction without a name that fits the domain"],
    },
    "synthesis": {
        "def": "the practice of combining ideas, evidence, or perspectives from multiple sources into a coherent integrated understanding that none of the sources individually provides",
        "principles": ["Read multiple sources before synthesizing", "Identify points of agreement and disagreement", "Look for higher-order patterns that reconcile apparent conflicts", "Cite the synthesis's lineage", "Mark inferences clearly"],
        "examples": ["Synthesizing three competing theories of a disease into a unified disease model", "Combining customer interview data with usage analytics into a single user model", "Merging two codebases into a unified architecture"],
        "mistakes": ["Confusing synthesis with summary", "Cherry-picking sources that agree", "Forcing reconciliation when sources genuinely conflict", "Presenting synthesis as the sources' view rather than your own", "Synthesizing without enough sources"],
    },
    "creative_thinking": {
        "def": "the practice of generating novel and useful ideas by combining existing elements in new ways, often by disrupting familiar patterns of association",
        "principles": ["Quantity breeds quality in idea generation", "Defer judgment during generation", "Combine distant concepts deliberately", "Constraints fuel creativity, they don't block it", "Rest and incubation matter — step away from the problem"],
        "examples": ["Combining a museum's exhibit design with game-design principles to create an interactive experience", "Applying jazz improvisation patterns to pair programming", "Designing a notification system inspired by how the brain gates sensory input"],
        "mistakes": ["Evaluating during generation", "Stopping at the first idea", "Treating constraints as enemies", "Confusing novelty with creativity (creativity requires usefulness)", "Forgetting that creative work needs rest periods"],
    },
}

REASONING_DEFS = {
    "deductive_reasoning": {
        "def": "reasoning from premises to a conclusion that follows necessarily if the premises are true; the conclusion is guaranteed by the form of the argument",
        "principles": ["Identify the inference form (modus ponens, modus tollens, etc.)", "Check validity before soundness", "Surface hidden premises", "Construct counterexamples to test validity", "Distinguish 'P→Q' from 'Q→P'"],
        "examples": ["All men are mortal. Socrates is a man. Therefore Socrates is mortal.", "If the build passes, we deploy. We did not deploy. Therefore the build did not pass.", "If P then Q. Not Q. Therefore not P."],
        "mistakes": ["Affirming the consequent", "Denying the antecedent", "Equivocating on terms", "Confusing validity with soundness", "Missing hidden premises"],
    },
    "inductive_reasoning": {
        "def": "reasoning from particular observations to a general conclusion that is probable but not guaranteed; the conclusion goes beyond the premises",
        "principles": ["Sample size matters, but representativeness matters more", "State the conclusion's scope honestly", "Watch for the black-swan refutation", "Distinguish 'all observed' from 'all'", "Update on new evidence"],
        "examples": ["Every swan I've seen is white, so all swans are white (refuted by black swans)", "1000 test cases pass, so the code is correct (refuted by case 1001)", "Every startup that raised Series A failed, so Series A causes failure (selection bias)"],
        "mistakes": ["Hasty generalization", "Cherry-picking observations", "Ignoring sample bias", "Treating inductive conclusions as certain", "Generalizing beyond the sample's representativeness"],
    },
    "abductive_reasoning": {
        "def": "inference to the best explanation — given an observation, generate the most plausible candidate explanations and rank them by explanatory power, simplicity, and plausibility",
        "principles": ["Generate multiple candidate explanations", "Rank by explanatory power × simplicity × plausibility", "Test in parallel where possible", "Don't commit to one until tested", "Watch for over-fitting to a single cause"],
        "examples": ["The app is slow for one user — generate 5 hypotheses (network, deploy, cache, extension, device) and rank", "Engagement dropped 20% after homepage launch — generate hypotheses (regression, tracking bug, competitor, seasonality)", "A bug appears intermittently — generate hypotheses (timing, state, external dependency)"],
        "mistakes": ["Locking onto the first plausible explanation", "Single-cause bias when multiple causes are likely", "Confusing plausibility with probability", "Failing to test alternatives in parallel", "Treating the best explanation as the only one"],
    },
    "analogical_reasoning": {
        "def": "reasoning that if two things are alike in known respects, they may be alike in further respects; the strength depends on the relevance of the similarities",
        "principles": ["Map the source domain to the target domain explicitly", "Identify the structural similarities that matter", "Check for disanalogies that break the inference", "Use analogy to generate hypotheses, not to confirm them", "Prefer deep structural analogies to surface analogies"],
        "examples": ["The heart is like a pump — generates questions about flow, pressure, valves", "Atoms are like solar systems — useful for teaching, breaks down for quantum behavior", "A code review is like a peer review of a paper — generates questions about criteria, bias, independence"],
        "mistakes": ["Stretching analogies past their domain", "Treating analogy as proof", "Choosing analogies by surface similarity rather than structural similarity", "Forgetting to check disanalogies", "Using analogies to manipulate rather than illuminate"],
    },
    "causal_reasoning": {
        "def": "reasoning about cause and effect — distinguishing causation from correlation, identifying causal mechanisms, and predicting the effects of interventions",
        "principles": ["Correlation is necessary but not sufficient for causation", "Look for the mechanism", "Test with intervention when possible", "Beware confounders", "Distinguish 'X causes Y' from 'X is associated with Y'"],
        "examples": ["Ice cream sales and drowning both rise in summer — confounded by temperature", "Users who use feature X retain better — but is X causing retention, or are retainers more likely to use X?", "Deploying on Fridays correlates with incidents — is Friday the cause, or are risky changes scheduled for Fridays?"],
        "mistakes": ["Confusing correlation with causation", "Ignoring confounders", "Reverse causation", "Common-cause fallacy", "Single-cause thinking when multiple causes are at work"],
    },
    "probabilistic_reasoning": {
        "def": "reasoning under uncertainty by explicitly representing and updating probabilities, rather than reasoning in certainties",
        "principles": ["Express beliefs as probabilities", "Update with Bayes' rule when evidence arrives", "Distinguish prior from posterior", "Calibrate probabilities against outcomes", "Avoid deterministic framing of probabilistic questions"],
        "examples": ["A test is 99% accurate; the disease rate is 1%; you test positive — what's the probability you have the disease? (Answer: 50%, not 99%)", "The model is 95% accurate on test set — what's the probability a specific prediction is correct? (Depends on base rate and confidence)", "Code review catches 80% of bugs; tests catch 90%; how many remain if both run? (Not 0%; depends on overlap)"],
        "mistakes": ["Base-rate neglect", "Confusing conditional probabilities (P(A|B) ≠ P(B|A))", "Overconfidence in single-event probabilities", "Treating probabilities as certainties", "Forgetting to update priors"],
    },
    "logical_fallacies": {
        "def": "common patterns of invalid reasoning that may appear persuasive on the surface but fail under examination",
        "principles": ["Name the fallacy to neutralize it", "Distinguish formal (structural) from informal (content) fallacies", "Identify the hidden assumption that makes the fallacy seem to work", "Don't accuse opponents of fallacies as a substitute for engaging", "Steel-man before fallacy-charging"],
        "examples": ["Ad hominem: 'You can't trust his code review — he's junior'", "Straw man: 'So you're saying we should ship without testing?'", "False dilemma: 'Either we ship now or we lose the market'"],
        "mistakes": ["Fallacy fallacy — dismissing a conclusion because the argument for it was fallacious", "False accusations of fallacy", "Confusing fallacy types", "Using fallacy labels as rhetorical weapons", "Forgetting that some arguments look like fallacies but aren't"],
    },
    "syllogisms": {
        "def": "a formal deductive argument with two premises and a conclusion, where the conclusion's subject and predicate are linked through a middle term",
        "principles": ["Identify the major, minor, and middle terms", "Check the figure and mood", "Test validity against the 24 valid syllogistic forms", "Beware the four perfect first-figure forms (Barbara, Celarent, Darii, Ferio)", "Watch for undistributed middle, illicit process"],
        "examples": ["Barbara: All M are P. All S are M. Therefore all S are P.", "Celarent: No M are P. All S are M. Therefore no S are P.", "Invalid: All cats are mammals. All dogs are mammals. Therefore all cats are dogs. (undistributed middle)"],
        "mistakes": ["Undistributed middle", "Illicit major/minor", "Exclusive premises (no valid conclusion)", "Affirmative conclusion from negative premise", "Existential fallacy (treating universal claims as existential)"],
    },
    "boolean_logic": {
        "def": "the algebra of true/false values and the operations (AND, OR, NOT, XOR, implication) that combine them",
        "principles": ["Apply De Morgan's laws to simplify negations", "Distribute AND over OR and vice versa", "Use truth tables to verify equivalence", "Distinguish implication (→) from biconditional (↔)", "Watch for operator precedence in expressions"],
        "examples": ["¬(A ∧ B) = ¬A ∨ ¬B (De Morgan)", "A → B = ¬A ∨ B (implication as disjunction)", "If (logged in AND admin) OR (logged in AND owner), then can edit = logged in AND (admin OR owner) (distribution)"],
        "mistakes": ["Confusing AND with OR in natural-language 'or'", "Negating implications incorrectly", "Forgetting operator precedence", "Confusing necessary with sufficient conditions", "Treating implication as biconditional"],
    },
    "modal_logic": {
        "def": "the logic of necessity and possibility — extending classical logic with operators for 'necessarily' (□) and 'possibly' (◇)",
        "principles": ["Distinguish alethic, deontic, epistemic, doxastic modalities", "Track the accessibility relation between possible worlds", "Watch for the duality □P = ¬◇¬P", "Different modal systems (K, T, S4, S5) for different applications", "Modal statements don't commute with negation"],
        "examples": ["□P (necessarily P): true in all accessible worlds", "◇P (possibly P): true in at least one accessible world", "K (knowledge): agent knows P; T (truth): what's known is true; S4 (positive introspection): if you know P, you know you know P"],
        "mistakes": ["Confusing □P with P", "Treating possibility as actuality", "Assuming all modal axioms apply (system choice matters)", "Forgetting the accessibility relation", "Modal scope errors ('necessarily, the author of X is Y' vs 'the author of X is necessarily Y')"],
    },
    "statistical_reasoning": {
        "def": "reasoning about collections of data using statistical concepts — distribution, variance, correlation, significance — while respecting their limitations",
        "principles": ["Always plot the data", "Distinguish sample from population", "Check distributional assumptions", "Effect size matters as much as significance", "Multiple comparisons require correction"],
        "examples": ["A/B test: p=0.04 — but is the effect size meaningful?", "Average salary up 10%, median down 5% — distribution shifted, not everyone got richer", "Correlation r=0.7 — depends on data range, sample, and outliers"],
        "mistakes": ["Confusing statistical significance with practical significance", "P-hacking", "Multiple comparisons without correction", "Drawing conclusions from non-representative samples", "Treating n=1 anecdotes as data"],
    },
    "bayesian_reasoning": {
        "def": "reasoning that updates the probability of a hypothesis as new evidence arrives, using Bayes' theorem: P(H|E) = P(E|H) P(H) / P(E)",
        "principles": ["State the prior explicitly", "Identify the likelihood (probability of evidence given hypothesis)", "Compute the marginal likelihood (probability of evidence under all hypotheses)", "Update incrementally as evidence accumulates", "Calibrate priors against long-run frequencies"],
        "examples": ["Disease test: 1% base rate, 99% sensitive, 99% specific → positive test = 50% chance of disease", "Code review finds a bug — what's the probability the code has more bugs? Depends on prior bug rate and review sensitivity", "Two competing models predict the same data; posterior odds = prior odds × Bayes factor"],
        "mistakes": ["Base-rate neglect", "Confusing P(H|E) with P(E|H)", "Using point estimates instead of distributions for priors", "Ignoring model uncertainty", "Updating only on confirmatory evidence"],
    },
    "counterfactual_reasoning": {
        "def": "reasoning about what would be true if some aspect of reality were different, used for causal attribution, moral judgment, and learning",
        "principles": ["Construct minimal counterfactuals — change one thing", "Use 'closest world' reasoning — the most similar world where the antecedent holds", "Distinguish counterfactual from causal claims", "Beware hindsight bias", "Use pre-mortems to generate forward-looking counterfactuals"],
        "examples": ["If the deploy had been delayed by 1 hour, would the outage still have happened?", "If we had hired the other candidate, would the project have shipped on time?", "If the user had read the warning, would they still have made the mistake?"],
        "mistakes": ["Hindsight bias", "Constructing implausible counterfactuals", "Multiple changes per counterfactual", "Treating counterfactual claims as observable facts", "Counterfactual cherry-picking"],
    },
    "diagnostic_reasoning": {
        "def": "reasoning from observed symptoms to underlying causes, often using a differential diagnosis process that ranks candidate causes by likelihood and testability",
        "principles": ["Generate a broad differential first", "Rank by likelihood × testability × seriousness", "Test to rule out, not just to confirm", "Update the differential as evidence arrives", "Don't anchor on the first plausible cause"],
        "examples": ["Patient with chest pain: differential includes muscle strain, reflux, anxiety, angina, PE; tests are ordered to rule out the serious ones", "Server slow: differential includes CPU, memory, disk I/O, network, app code, database; monitor each", "User reports 'broken': differential includes user error, browser, network, app, account state"],
        "mistakes": ["Premature closure on first plausible cause", "Anchoring on the salient symptom", "Failure to consider serious-but-rare causes", "Testing only for confirmation, not exclusion", "Forgetting to update after negative tests"],
    },
    "spatial_reasoning": {
        "def": "reasoning about the positions, shapes, and relationships of objects in space — including mental rotation, perspective-taking, and 3D visualization",
        "principles": ["Use diagrams when possible", "Practice mental rotation explicitly", "Take multiple perspectives (top, side, internal)", "Track relative vs absolute position", "Use coordinate systems deliberately"],
        "examples": ["Reading a circuit board layout to trace signal paths", "Visualizing a 3D object from its 2D projections", "Reasoning about cache locality in data layout"],
        "mistakes": ["Confusing relative and absolute position", "Forgetting the third dimension in 2D plans", "Mirror-image confusion in mental rotation", "Anchoring on a single perspective", "Underestimating distances or sizes"],
    },
    "temporal_reasoning": {
        "def": "reasoning about the ordering, duration, and concurrency of events in time, including before/after relationships and causal timing",
        "principles": ["Draw timelines explicitly", "Distinguish 'before' from 'because of'", "Watch for delays between cause and effect", "Account for time zones and clocks", "Reason about concurrency explicitly"],
        "examples": ["Reading a distributed-system trace to find the first failure", "Reasoning about whether an event caused or merely preceded another", "Scheduling dependent tasks with explicit timing constraints"],
        "mistakes": ["Post hoc ergo propter hoc (after this therefore because of this)", "Forgetting timezone conversions", "Ignoring clock skew in distributed systems", "Confusing 'concurrent' with 'simultaneous'", "Overlooking delays in feedback loops"],
    },
    "moral_reasoning": {
        "def": "reasoning about what one ought to do, given consideration of values, duties, consequences, and the perspectives of those affected",
        "principles": ["Identify the moral framework(s) in play (consequentialist, deontological, virtue)", "Consider the perspectives of all affected parties", "Distinguish moral claims from factual claims", "Watch for moral dumbfounding — intuitions without reasons", "Distinguish 'is' from 'ought'"],
        "examples": ["Should you lie to protect a colleague from an unjust punishment? (deontological 'no' vs consequentialist 'yes')", "Is it permissible to use data collected for one purpose for another? (consent-based reasoning)", "Should an AI be truthful even when truth causes harm?"],
        "mistakes": ["Confusing moral intuition with moral argument", "Moral equivalence fallacy", "Appeal to nature ('it's natural, therefore good')", "Confusing legal with moral", "Forgetting the affected parties who aren't in the room"],
    },
    "ethical_dilemmas": {
        "def": "situations where two or more moral principles conflict and no choice satisfies all of them, requiring explicit weighing of competing values",
        "principles": ["Identify the competing principles", "Consider all stakeholders", "Distinguish 'right vs right' from 'right vs wrong'", "Document the reasoning for review", "Acknowledge the residual wrong in any choice"],
        "examples": ["Trolley problem: divert to save 5 at the cost of 1", "Whistleblowing: loyalty to employer vs duty to public", "Triage: allocate limited medical resources among patients"],
        "mistakes": ["Pretending the dilemma isn't real", "Choosing by gut without examining principles", "Forgetting stakeholders not present", "Treating the dilemma as solvable rather than navigable", "Failing to document the reasoning"],
    },
    "counterargument_construction": {
        "def": "the practice of building the strongest possible argument against one's own position, to test its strength and anticipate objections",
        "principles": ["Steel-man the opposing position", "Find the strongest version, not the easiest to refute", "Acknowledge the points where the opposition is right", "Identify the empirical question that would settle the dispute", "Avoid strawman versions"],
        "examples": ["If you believe 'remote work hurts collaboration,' construct the strongest case that it doesn't (and check it against data)", "If you believe 'microservices are better,' construct the strongest case for monoliths in this context", "If you believe 'we should ship now,' construct the strongest case for waiting"],
        "mistakes": ["Strawman construction", "Cherry-picking the weakest opposing points", "Forgetting to test the steel-manned version against data", "Constructing counterarguments only to dismiss them", "Confusing steel-manning with agreement"],
    },
    "proof_verification": {
        "def": "the practice of checking a logical or mathematical proof step by step, verifying each inference and identifying any gaps or errors",
        "principles": ["Check each step's inference form", "Verify definitions are used consistently", "Look for hidden assumptions", "Check edge cases (empty set, base case, etc.)", "Independent re-derivation beats re-reading"],
        "examples": ["Verifying a mathematical induction proof: base case + inductive step", "Verifying a code-correctness proof: preconditions, invariants, postconditions", "Verifying a security proof: assumptions, threat model, reduction"],
        "mistakes": ["Skimming instead of step-checking", "Trusting familiar patterns instead of verifying", "Missing edge cases", "Confusing 'I can't find an error' with 'the proof is correct'", "Forgetting to verify the assumptions hold in the application"],
    },
}

SPEAKING_DEFS = {
    "clear_expression": {
        "def": "the practice of conveying intended meaning with minimal noise, ambiguity, or effort required on the receiver's part",
        "principles": ["Prefer verbs to nominalizations", "Prefer concrete numbers to vague intensifiers", "Prefer active voice unless the actor is irrelevant", "Cut every word the sentence survives without", "Read aloud; if it doesn't flow, rewrite"],
        "examples": ["'The team's fix cut p99 latency in half' vs 'the implementation of the solution resulted in a significant enhancement'", "'Ship Tuesday' vs 'we should possibly consider the feasibility of an imminent deployment'", "'Three customers complained about X' vs 'some customers had concerns'"],
        "mistakes": ["Stacking hedges ('maybe possibly sort of try')", "Burying the lede", "Using jargon the audience doesn't share", "Confusing formal-sounding with clear", "Filling with throat-clearing openers"],
    },
    "structured_communication": {
        "def": "the practice of organizing a message so the receiver can navigate it efficiently — bottom-line up front, supporting details after, signposted throughout",
        "principles": ["BLUF: bottom line up front", "Signpost structure ('three points: ...')", "One idea per paragraph", "Headers for any message > 3 paragraphs", "End with the ask or next step"],
        "examples": ["Exec memo: 1-sentence recommendation, 3 supporting bullets, 1 risk, 1 ask", "Slack update: status (green/yellow/red), one-sentence summary, blockers if any, ask if any", "Engineering design doc: problem, proposed solution, alternatives considered, tradeoffs, open questions"],
        "mistakes": ["Building up to the conclusion", "Multiple ideas per paragraph", "No signposts in long messages", "Asking without stating what you need", "Forgetting to mark what's a decision vs a discussion"],
    },
    "persuasion": {
        "def": "the practice of changing someone's beliefs, attitudes, or actions through reasoned argument, evidence, and emotional appeal — without manipulation",
        "principles": ["Start with shared common ground", "Address the strongest counterargument first", "Use concrete examples over abstractions", "Match evidence type to audience (data vs story)", "End with a clear, low-friction ask"],
        "examples": ["Persuading leadership to invest in observability: lead with the cost of the last outage, not with the abstract value of monitoring", "Persuading a team to adopt tests: tell the story of the bug that tests would have caught", "Persuading a customer to upgrade: focus on the specific problem the upgrade solves, not the feature list"],
        "mistakes": ["Leading with disagreement", "Ignoring the strongest counterargument", "Using data when story is needed (or vice versa)", "Vague asks ('be more thoughtful') vs concrete ones ('review my draft by Friday')", "Confusing persuasion with coercion"],
    },
    "rhetoric": {
        "def": "the art of effective communication, especially persuasive communication, using ethos (credibility), pathos (emotion), and logos (logic) in balanced proportion",
        "principles": ["Establish ethos early", "Use pathos sparingly and authentically", "Anchor in logos for the substantive argument", "Vary sentence rhythm for emphasis", "End on the strongest point"],
        "examples": ["A keynote that opens with a personal story (pathos), cites the speaker's track record (ethos), then walks through the technical argument (logos)", "A fundraising pitch that opens with the founder's credibility, evokes the user's pain, then presents the numbers", "A code-review comment that establishes the reviewer's stake, names the cost of the bug, then walks through the fix"],
        "mistakes": ["Pathos without logos (manipulation)", "Logos without ethos (won't be heard)", "Ethos without either (posturing)", "Flat sentence rhythm", "Ending on the weakest point"],
    },
    "storytelling": {
        "def": "the practice of structuring information as a narrative — with characters, stakes, conflict, and resolution — to make it memorable and emotionally resonant",
        "principles": ["Open with a hook that establishes stakes", "Give the audience a character to identify with", "Build tension through obstacles", "Resolution should feel earned, not convenient", "Concrete details beat abstractions"],
        "examples": ["Customer story: a specific user, a specific problem, a specific moment of resolution", "Incident retrospective as a story: the team, the warning signs, the crisis, the recovery, the lesson", "Founding story: the moment of recognition, the bet, the obstacles, the breakthrough"],
        "mistakes": ["No stakes (boring)", "Abstraction instead of character", "Convenient resolution (unconvincing)", "Telling instead of showing", " burying the hook in background"],
    },
    "technical_explanation": {
        "def": "the practice of explaining technical concepts to audiences of varying technical depth, calibrating the level of abstraction and the use of analogy to the receiver",
        "principles": ["Calibrate to the receiver's existing mental model", "Use analogy to bridge from known to unknown", "Worked example > abstract description", "Name jargon on first use; don't ban it", "Check understanding by asking the receiver to restate"],
        "examples": ["Explaining distributed consensus to a product manager using the 'team agreeing on lunch' analogy", "Explaining recursion to a junior dev with factorial and a call-stack trace", "Explaining backpressure to a senior engineer with the leaky-bucket model"],
        "mistakes": ["Assuming the receiver's mental model", "Analogy that breaks the concept", "All abstraction, no example", "Jargon without definition", "No comprehension check"],
    },
    "executive_communication": {
        "def": "the practice of communicating to senior leaders who have limited time, broad scope, and decision-making authority — requiring crisp framing, options, and recommendations",
        "principles": ["Lead with the recommendation", "Provide 2-3 options with tradeoffs", "Quantify impact and risk", "Specify the ask and the decision needed", "Pre-empt the obvious questions"],
        "examples": ["Memo to CEO: 'Recommend we acquire X for $Y. Strategic rationale: ... Alternatives considered: ... Risk: ... Decision needed by: ...'", "Update to the board: 'Q3 on track to beat plan by 8%. Two items needing board input: ...'", "CFO escalation: 'Need approval for $Z unplanned spend. Cause, impact of not spending, recommended action.'"],
        "mistakes": ["Building up to the recommendation", "Presenting one option (no real decision)", "Vague impact ('it'll be better')", "No clear ask", "Not pre-empting the obvious questions"],
    },
    "negotiation": {
        "def": "the practice of reaching an agreement with another party whose interests partly overlap and partly conflict, through structured exchange of information and proposals",
        "principles": ["Separate people from the problem", "Focus on interests, not positions", "Invent options for mutual gain", "Insist on objective criteria", "Know your BATNA before starting"],
        "examples": ["Salary negotiation: research market rates, frame around shared interest in retention, propose objective criteria", "Vendor negotiation: separate the relationship from the deal, find non-price levers (term length, payment timing)", "Cross-team priority negotiation: surface each team's underlying constraints, find an option that addresses both"],
        "mistakes": ["Anchoring on your own number first", "Confusing position with interest", "Forgetting your BATNA", "Treating negotiation as zero-sum", "Letting emotion drive concessions"],
    },
    "conflict_resolution": {
        "def": "the practice of resolving disagreements between parties by addressing the underlying interests and restoring working relationships, not just settling the surface dispute",
        "principles": ["Surface the underlying interests, not just positions", "Use 'I' statements, not 'you' statements", "Separate the problem from the people", "Look for tradeables across issues", "Rebuild the relationship explicitly afterward"],
        "examples": ["Two engineers disagree on architecture: surface each one's underlying concern (perf vs maintainability), find a design that addresses both", "Team conflict over priority: surface the underlying constraints, find a sequence that addresses both", "Personal conflict: name the behavior, not the person; specify the impact; propose a different behavior"],
        "mistakes": ["Focusing on positions instead of interests", "Personal attacks", "Avoiding the conflict until it festers", "Imposing a resolution without buy-in", "Forgetting to repair the relationship after"],
    },
    "active_listening": {
        "def": "the practice of fully attending to a speaker, understanding their meaning, and reflecting it back — rather than planning your response while they talk",
        "principles": ["Attend fully — put down the phone", "Don't plan your response while they're speaking", "Reflect back what you heard in your own words", "Ask clarifying questions, not leading ones", "Notice nonverbal cues"],
        "examples": ["'So what I'm hearing is that you're concerned about the timeline — is that right?'", "'Let me make sure I understand: you're saying X because of Y. Did I get that?'", "Team member: 'I'm overwhelmed.' Active response: 'Tell me what's on your plate — let's see what we can move.'"],
        "mistakes": ["Planning your response while they talk", "Interrupting with your own story", "Giving advice when empathy was needed", "Reflecting back without paraphrasing", "Missing the emotion underneath the words"],
    },
    "feedback_delivery": {
        "def": "the practice of giving feedback that the receiver can hear, accept, and act on — by being specific, behavioral, timely, and balanced",
        "principles": ["Specific, not general ('you interrupted three times in the meeting' not 'you're rude')", "Behavioral, not character ('when you did X' not 'you are Y')", "Timely — give it soon after the event", "Balance corrective with reinforcing", "Ask, don't assume, the receiver's intent"],
        "examples": ["'In yesterday's review, three of your comments used sarcasm. The author stopped responding afterward. I'd love to understand what was going on for you.'", "'Your doc was crystal clear on the problem statement. One suggestion: the alternatives section could use one more option.'", "'In the standup, you talked over Priya twice. I noticed she didn't speak again after.'"],
        "mistakes": ["Vague ('you need to be more professional')", "Character attacks ('you're lazy')", "Delayed past usefulness", "All corrective, no reinforcing", "Assuming intent ('you did this because...')"],
    },
    "public_speaking": {
        "def": "the practice of delivering a presentation to a live audience with structure, engagement, and clear takeaway — managing both content and delivery",
        "principles": ["Open with a hook", "Have one big idea", "Use stories to anchor concepts", "Vary pacing and tone", "End with a call to action or memorable line"],
        "examples": ["Conference talk: open with a story, present one model, give three applications, end with a question", "All-hands: state the headline, walk through the data, acknowledge what's hard, end with what's next", "Lightning talk: one problem, one insight, one action"],
        "mistakes": ["Reading slides", "No clear takeaway", "Pacing too fast or too slow", "Ignoring the audience's energy", "Endings that trail off"],
    },
    "interview_skills": {
        "def": "the practice of conducting or participating in interviews — whether journalistic, hiring, or research — to elicit high-quality information",
        "principles": ["Prepare, don't script", "Open with broad questions, narrow down", "Ask one question at a time", "Silence is a tool — let them fill it", "Probe for specifics, not generalities"],
        "examples": ["Hiring: 'Tell me about a time you shipped something hard.' → 'What made it hard?' → 'What did you do specifically?'", "User research: 'Walk me through the last time you used X.' → 'What were you thinking at that moment?'", "Journalistic: 'What happened?' → 'Then what?' → 'Why do you think that?'"],
        "mistakes": ["Leading questions", "Compound questions", "Filling silences", "Accepting vague answers without probing", "Talking more than listening"],
    },
    "explanatory_teaching": {
        "def": "the practice of explaining a concept so the learner builds a usable mental model — not just memorizes a definition",
        "principles": ["Start from what the learner already knows", "Use analogies that map structurally", "Worked examples before abstract rules", "Check understanding by asking the learner to apply, not restate", "Surface and address misconceptions explicitly"],
        "examples": ["Teaching recursion: 'Remember how Russian nesting dolls work? Each doll contains a smaller one until you hit the tiniest.' Then show factorial with a call-stack trace", "Teaching Bayes' theorem: 'Imagine 1000 people. 10 have the disease. Of those, 9 test positive. Of the 990 without, ~10 test positive. So 9/(9+10) ≈ 47%.' Then derive the formula", "Teaching CAP: 'You're at a restaurant. The kitchen can be either fast or accurate but not both. Why?' Then formalize"],
        "mistakes": ["Starting from the formal definition", "Analogy that doesn't map", "Abstraction before example", "Confusing 'I explained it' with 'they understood it'", "Skipping misconceptions"],
    },
    "concise_writing": {
        "def": "the practice of expressing ideas in the fewest words that convey the full meaning — without losing nuance or precision",
        "principles": ["One idea per sentence", "Cut every word the sentence survives without", "Prefer short words to long when meaning is equal", "Prefer active voice", "Read aloud and cut where you stumble"],
        "examples": ["'Now' instead of 'at this point in time'", "'Because' instead of 'due to the fact that'", "'To' instead of 'in order to'", "'The team shipped the fix' instead of 'the fix was successfully shipped by the team'"],
        "mistakes": ["Padding for formality", "Adverbs that duplicate the verb's meaning", "Hedge stacking", "Long words when short ones work", "Compound sentences that hide the main point"],
    },
    "diplomacy": {
        "def": "the practice of communicating sensitive information in a way that preserves the relationship while still conveying the message",
        "principles": ["Separate the message from the framing", "Choose the right channel (private for hard messages)", "Lead with shared interest", "Name the behavior, not the person", "Offer a path forward"],
        "examples": ["Telling a peer their work needs rework: 'I want to make sure this lands well for you. Can we walk through it together? I have some concerns about X.'", "Telling leadership a deadline will slip: 'We've hit a snag in X. I want to flag it early. Here's the impact, here's what we're trying, here's what we need.'", "Declining a request: 'I want to help. Right now my plate is X. Can we de-prioritize Y to make room?'"],
        "mistakes": ["Softening the message so much it's lost", "Public channels for sensitive feedback", "Blaming the person instead of the behavior", "No path forward", "Diplomacy as avoidance"],
    },
    "question_formulation": {
        "def": "the practice of asking questions that elicit the information you actually need — open when exploring, closed when confirming",
        "principles": ["Open questions to explore ('what', 'how', 'tell me about')", "Closed questions to confirm ('is it X or Y?')", "Avoid leading questions", "Avoid compound questions", "Sequence: open → narrow → confirm"],
        "examples": ["Instead of 'Don't you think we should ship?' → 'What are the tradeoffs of shipping now vs waiting?'", "Instead of 'Was it the deploy that broke it?' → 'What changed around the time it broke?'", "Instead of 'Are you happy with the plan and the timeline?' → 'What's your reaction to the plan?' then separately 'How's the timeline feel?'"],
        "mistakes": ["Leading questions", "Compound questions", "Yes/no questions when you need exploration", "Open questions when you need confirmation", "Questions that signal the answer you want"],
    },
    "analogy_crafting": {
        "def": "the practice of constructing analogies that map structurally from a known domain to an unfamiliar one, transferring understanding without distortion",
        "principles": ["Map the structural relations, not the surface features", "Verify the analogy's predictions in the target domain", "Acknowledge where the analogy breaks", "Use multiple analogies for the same concept to triangulate", "Match the analogy to the receiver's prior knowledge"],
        "examples": ["Cache is like a refrigerator — small, fast, close; main memory is like the supermarket — big, slow, far", "Hash tables are like a coat check — you give a number (hash), get back your coat (value)", "Distributed consensus is like a jury trying to reach a verdict — they need a protocol that guarantees agreement even if some members are dishonest"],
        "mistakes": ["Surface analogies that don't transfer", "Stretching the analogy past its domain", "Single analogy used as proof", "Analogy that misleads about the hard parts", "Forgetting to mark where it breaks"],
    },
    "audience_adaptation": {
        "def": "the practice of adjusting the level, tone, examples, and structure of a message to the specific audience receiving it",
        "principles": ["Know the audience's prior knowledge", "Know the audience's stakes", "Choose examples from the audience's domain", "Adjust jargon density to audience fluency", "Match the channel to the audience's preference"],
        "examples": ["Same incident: to engineers, deep technical detail; to executives, business impact and customer impact; to customers, what happened, what we're doing, what to expect", "Same feature: to designers, the user flows; to engineers, the data model and APIs; to sales, the value props and objections", "Same code change: to the PR reviewer, the diff and tests; to the team, the rationale and impact; to the on-call, the operational notes"],
        "mistakes": ["One-size-fits-all messaging", "Executive detail to engineers (boring) or engineer detail to executives (overwhelming)", "Wrong jargon level", "Wrong examples for the audience", "Wrong channel"],
    },
    "tone_calibration": {
        "def": "the practice of matching the emotional register of a message to the situation, the audience, and the desired effect",
        "principles": ["Match urgency to actual urgency", "Match formality to the relationship", "Acknowledge emotion before content", "Choose warmth vs neutrality deliberately", "Watch for tone mismatch (casual in crisis, formal in friendship)"],
        "examples": ["Crisis message: 'We have an active outage. Customers affected: ~5%. Mitigation underway. Next update in 30 min.' (urgent, factual, calm)", "Correction to a peer: 'Quick note: I think there might be an off-by-one in the loop. Want me to send a fix?' (warm, low-stakes)", "Serious concern to leadership: 'I want to flag a risk I think we're underweighting. Happy to discuss when you have time.' (formal, measured)"],
        "mistakes": ["Casual tone in a serious situation", "Formal tone in a casual relationship", "All-caps / exclamation inflation", "No acknowledgement of emotion in charged moments", "Tone that doesn't match the stakes"],
    },
}

UNDERSTANDING_DEFS = {
    "reading_comprehension": {
        "def": "the practice of extracting the literal meaning, the implied meaning, and the author's intent from a written text",
        "principles": ["Read actively — ask questions as you go", "Distinguish main claim from supporting evidence", "Track the argument's structure", "Note unfamiliar terms and look them up", "Summarize at the end of each section"],
        "examples": ["Reading a research paper: identify the claim, the method, the data, the limitations", "Reading a contract: identify obligations, conditions, termination clauses, indemnities", "Reading a design doc: identify problem, proposed solution, alternatives, tradeoffs, open questions"],
        "mistakes": ["Passive reading without questioning", "Confusing summary with understanding", "Skipping unfamiliar terms", "Missing the argument's structure", "Forgetting the author's intent"],
    },
    "literal_interpretation": {
        "def": "the practice of extracting the surface meaning of a text — what the words actually say, independent of what they might imply",
        "principles": ["Read each sentence for what it says, not what you assume it means", "Distinguish 'said' from 'implied'", "Note qualifiers (some, all, often, never) carefully", "Track pronoun referents explicitly", "Don't import context the text doesn't provide"],
        "examples": ["'Some users prefer the old version' → not 'all users' nor 'most users'", "'The feature is supported on Chrome' → says nothing about Firefox, Safari, or other browsers", "'We will review the request' → commitment to review, not to approve"],
        "mistakes": ["Reading implications as facts", "Over-generalizing qualified claims", "Assuming pronoun referents without checking", "Importing unstated context", "Treating 'not mentioned' as 'not true'"],
    },
    "inferential_comprehension": {
        "def": "the practice of drawing conclusions that go beyond the literal text — what must be true for the text to make sense, or what the author likely meant but didn't state",
        "principles": ["Distinguish what the text says from what it implies", "Mark inferences as inferences, not as text", "Check the inference against the text — does it contradict?", "Watch for inferences that import outside assumptions", "Calibrate confidence: 'must be true' vs 'probably true' vs 'could be true'"],
        "examples": ["'The meeting ran long. The CEO left early.' → likely the CEO had a conflict, but not stated", "'Sales grew 20%. The competitor cut prices.' → likely the growth was despite the price cut, but causal direction isn't stated", "'The code has no tests.' → likely brittle, but not stated"],
        "mistakes": ["Treating inferences as text", "Importing outside assumptions as inferences", "Over-claiming confidence in the inference", "Drawing inferences the text doesn't support", "Confusing 'could be true' with 'must be true'"],
    },
    "contextual_understanding": {
        "def": "the practice of interpreting a message in light of its surrounding context — historical, cultural, situational, and relational",
        "principles": ["Identify the context the message was produced in", "Consider the speaker's role, history, and stake", "Account for the situational stakes", "Watch for cultural context that shifts meaning", "Distinguish universal from context-bound meaning"],
        "examples": ["'Interesting approach' in a code review means 'I have concerns' — in a brainstorm it means 'tell me more'", "'We should do lunch sometime' in the US is often a polite brush-off; in some cultures it's a real invitation", "'That's one way to do it' from a senior engineer is rarely neutral"],
        "mistakes": ["Stripping context to read literally", "Assuming one's own context is universal", "Missing power dynamics in the message", "Forgetting cultural context", "Confusing situational with dispositional"],
    },
    "perspective_taking": {
        "def": "the practice of understanding a situation from another person's point of view — their beliefs, goals, constraints, and emotions",
        "principles": ["Name the other's likely goals and constraints", "Consider what they know that you don't (and vice versa)", "Watch for information asymmetry", "Imagine the message from their position", "Distinguish understanding from agreement"],
        "examples": ["Before pushing back on a stakeholder's request, articulate their underlying goal in one sentence", "Before responding to a critical code review, restate the reviewer's concern in your own words", "Before escalating to leadership, anticipate what they care about (risk, timeline, customer impact)"],
        "mistakes": ["Confusing perspective-taking with agreement", "Projecting your goals onto the other", "Forgetting information asymmetry", "Treating the other's view as irrational", "Perspective-taking as a one-time exercise, not ongoing"],
    },
    "empathy": {
        "def": "the practice of attending to and validating another's emotional experience, separately from one's own emotional response to it",
        "principles": ["Listen for the emotion underneath the words", "Reflect the emotion back ('that sounds frustrating')", "Validate before problem-solving", "Separate your emotional reaction from theirs", "Don't conflate empathy with agreement"],
        "examples": ["Team member vents about a hard week: 'That sounds exhausting. Do you want to vent, or do you want to problem-solve?'", "Customer angry about a bug: 'I can hear how disruptive this has been. I'm sorry. Here's what we're doing.'", "Peer upset about a decision: 'I can see why that's disappointing. Tell me more about what's hardest about it.'"],
        "mistakes": ["Jumping to problem-solving before validating", "Dismissing the emotion as disproportionate", "Making it about yourself ('I had that too!')", "Confusing empathy with agreement", "Empathy without follow-through"],
    },
    "metaphor_interpretation": {
        "def": "the practice of recognizing when language is being used figuratively and recovering the intended meaning, rather than reading it literally",
        "principles": ["Notice when literal meaning is implausible or absurd", "Identify the source domain and the target domain", "Map the structural relations", "Watch for dead metaphors (no longer recognized as figurative)", "Distinguish metaphor from analogy and idiom"],
        "examples": ["'The project is on fire' → not literally burning; means in serious trouble", "'She has a heart of stone' → not literally stone; means emotionally unresponsive", "'The codebase is spaghetti' → not literally pasta; means tangled and hard to follow"],
        "mistakes": ["Reading figurative language literally", "Forcing an interpretation when the metaphor is mixed", "Confusing metaphor with analogy (analogy is explicit, metaphor is implicit)", "Treating dead metaphors as living", "Stretching the metaphor past the author's intent"],
    },
    "ambiguity_resolution": {
        "def": "the practice of identifying when a message has multiple possible meanings and using context, priors, and clarification to settle on the intended one",
        "principles": ["Identify the alternative interpretations", "Use context to rank them", "Use the speaker's likely intent to rank further", "Ask for clarification when stakes are high", "Mark unresolved ambiguity explicitly"],
        "examples": ["'I saw the man with the telescope' — who has the telescope?", "'Flying planes can be dangerous' — flying as gerund or adjective?", "'The team is ready to ship' — ready as in 'about to' or 'willing to'?"],
        "mistakes": ["Locking on the first interpretation without checking alternatives", "Treating ambiguity as the speaker's error", "Failing to clarify when stakes are high", "Pretending ambiguity doesn't exist", "Confusing ambiguity with vagueness"],
    },
    "implicit_meaning": {
        "def": "the practice of recognizing what a message implies but does not state — the presuppositions, implicatures, and unstated conclusions the speaker expects the receiver to draw",
        "principles": ["Distinguish what's said from what's presupposed", "Identify conversational implicatures (Gricean maxims)", "Watch for presuppositions embedded in questions", "Track what the speaker assumes you know", "Mark implicatures as inferences, not statements"],
        "examples": ["'Have you stopped working from home?' presupposes you used to", "'Some students passed the test' implicates (but doesn't say) not all did", "'The food was OK' implicates 'not great'"],
        "mistakes": ["Treating implicatures as explicit claims", "Missing presuppositions entirely", "Confusing entailment with implicature", "Drawing implicatures the speaker didn't intend", "Treating 'not stated' as 'not implied'"],
    },
    "cultural_context": {
        "def": "the practice of interpreting a message in light of the cultural background of the speaker and the situation — recognizing that meaning is partly culture-bound",
        "principles": ["Identify the speaker's cultural frame", "Watch for cultural idioms and references", "Account for power-distance and formality norms", "Recognize culture-bound politeness strategies", "Avoid assuming one's own cultural defaults"],
        "examples": ["'Maybe' in some cultures means 'no, politely' — not 'perhaps'", "Direct negative feedback is normal in some cultures, deeply offensive in others", "Silence can mean 'thinking,' 'agreement,' or 'disagreement' depending on culture"],
        "mistakes": ["Treating cultural defaults as universal", "Missing cultural references", "Confusing cultural norms with personal style", "Over-correcting (treating all behavior as cultural)", "Forgetting subcultures within a culture"],
    },
    "discourse_analysis": {
        "def": "the practice of analyzing stretches of language longer than a sentence — how the parts connect, how coherence is achieved, and how the discourse functions as a whole",
        "principles": ["Identify the discourse structure (problem-solution, claim-evidence, narrative)", "Track cohesive devices (pronouns, conjunctions, lexical chains)", "Watch for topic shifts and boundaries", "Identify the discourse's social function", "Note the rhetorical moves the speaker makes"],
        "examples": ["A research paper: intro (problem), methods (how), results (what), discussion (so what)", "A political speech: identify the framing, the appeals, the call to action", "A PR description: problem, approach, alternatives, impact"],
        "mistakes": ["Analyzing sentences in isolation", "Missing the discourse structure", "Confusing cohesion with coherence", "Forgetting the social function", "Treating discourse as just sentences"],
    },
    "summarization": {
        "def": "the practice of producing a shorter version of a text that preserves the essential information while omitting detail",
        "principles": ["Identify the main claim", "Include the key supporting evidence", "Omit illustrative detail", "Preserve the original's stance", "Match length to purpose"],
        "examples": ["One-paragraph summary of a 20-page paper: main claim + method + key finding + limitation", "One-sentence summary of an incident: what broke, for whom, for how long, why", "Five-bullet exec summary of a 60-minute meeting: decisions made, action items, owners"],
        "mistakes": ["Including too much detail", "Missing the main claim", "Distorting the original's stance", "Inappropriate length for the purpose", "Confusing summary with critique"],
    },
    "synthesis_across_sources": {
        "def": "the practice of integrating information from multiple sources into a unified understanding that resolves conflicts and identifies higher-order patterns",
        "principles": ["Read all sources before synthesizing", "Identify points of agreement", "Identify points of disagreement and the reasons", "Look for higher-order patterns that reconcile", "Cite the synthesis's lineage"],
        "examples": ["Synthesizing three user-research reports into a unified user model", "Synthesizing competing technical analyses of an incident into a single timeline", "Synthesizing multiple forecasts into a probability distribution"],
        "mistakes": ["Cherry-picking sources that agree", "Forcing reconciliation when sources genuinely conflict", "Confusing synthesis with average", "Presenting synthesis as the sources' view", "Forgetting to mark inferences"],
    },
    "fact_checking": {
        "def": "the practice of verifying factual claims against authoritative sources before accepting or repeating them",
        "principles": ["Identify the specific claim", "Find the authoritative source", "Check the claim against the source", "Distinguish the original source from secondary reporting", "Note when verification fails"],
        "examples": ["A viral statistic — find the original study, check the methodology, check the sample size", "A quote attributed to a famous person — find the original source, check the context", "A historical claim — find the primary source, not the textbook"],
        "mistakes": ["Trusting secondary sources as primary", "Confusing a fact check with a Google search", "Verification theater (looking without rigor)", "Forgetting to check the date", "Treating 'no disconfirming evidence' as 'confirmed'"],
    },
    "source_evaluation": {
        "def": "the practice of assessing the credibility, expertise, and potential bias of an information source",
        "principles": ["Identify the source's expertise on this specific topic", "Check for conflicts of interest", "Assess the source's track record", "Distinguish institutional from individual credibility", "Watch for self-citation patterns"],
        "examples": ["A medical claim from a cardiologist vs a generalist vs a wellness blogger", "A financial claim from a fund manager vs an academic vs a journalist", "A technical claim from a maintainer vs a user vs a journalist"],
        "mistakes": ["Treating all credentials as equal", "Ignoring conflicts of interest", "Confusing fame with expertise", "Treating consensus as proof", "Treating outlier expertise as authoritative"],
    },
    "intention_recognition": {
        "def": "the practice of inferring what a speaker is trying to accomplish with a message — beyond the literal content",
        "principles": ["Identify the speech act (asserting, requesting, warning, promising, etc.)", "Consider the speaker's stakes", "Watch for indirect speech acts", "Distinguish information-seeking from validation-seeking", "Note the audience the speaker is really addressing"],
        "examples": ["'Is it cold in here?' may be a request to close the window, not a temperature question", "'You look tired' may be concern, criticism, or a request to take a break", "'Are you sure?' may be a request to reconsider, not a question about confidence"],
        "mistakes": ["Treating indirect speech as literal", "Missing the speech act entirely", "Projecting one's own intentions onto the speaker", "Confusing the surface audience with the real audience", "Assuming intention is singular"],
    },
    "subtext_detection": {
        "def": "the practice of recognizing what's left unsaid but is nonetheless part of the message — the meaning carried by absence, emphasis, and selection",
        "principles": ["Note what's not said", "Watch for unusual emphasis", "Track what's selected for mention vs omitted", "Consider the speaker's stake in the subtext", "Mark subtext as inference, not statement"],
        "examples": ["'The new hire is... enthusiastic' — the pause carries the meaning", "A performance review that lists achievements but no growth areas — the absence is the message", "'I'll keep that in mind' often means 'I will not act on this'"],
        "mistakes": ["Treating subtext as text", "Over-reading when there's nothing there", "Missing subtext entirely", "Confusing subtext with implication", "Projecting one's own concerns as subtext"],
    },
    "pragmatic_understanding": {
        "def": "the practice of interpreting language in light of how it's actually used in context — including politeness strategies, indirectness, and conversational norms",
        "principles": ["Apply Grice's maxims (quantity, quality, relation, manner)", "Recognize politeness strategies (off-record, negative, positive)", "Account for indirectness norms", "Watch for face-threatening acts and how they're mitigated", "Distinguish what's said from what's done"],
        "examples": ["'Could you possibly...?' is a polite request, not a question about ability", "'I was wondering if...' is a hedge, not actual wonder", "'Some might say...' is a way to make a claim without owning it"],
        "mistakes": ["Reading pragmatic indirectness as literal", "Missing the speech act being performed", "Confusing politeness with agreement", "Treating indirectness as deception", "Forgetting cultural pragmatic norms"],
    },
    "concept_extraction": {
        "def": "the practice of identifying the key concepts in a text and the relationships among them",
        "principles": ["Identify the central concepts", "Identify the peripheral concepts", "Map the relationships (is-a, has-a, causes, depends-on)", "Note the level of abstraction", "Watch for concepts that are assumed but not named"],
        "examples": ["Extracting concepts from a design doc: problem, solution, alternatives, tradeoffs, constraints, assumptions", "Extracting concepts from a research paper: hypothesis, method, variables, controls, findings", "Extracting concepts from a contract: parties, obligations, conditions, termination, remedies"],
        "mistakes": ["Confusing concepts with terms", "Missing implicit concepts", "Treating all concepts as equally central", "Missing the relationships", "Confusing level of abstraction"],
    },
    "semantic_relationships": {
        "def": "the practice of recognizing the relationships between words and concepts — synonymy, antonymy, hypernymy, meronymy, and entailment",
        "principles": ["Distinguish synonymy (same meaning) from relatedness", "Identify antonymy (gradable vs complementary vs relational)", "Recognize hypernymy (is-a) and meronymy (part-of)", "Track entailment (P entails Q)", "Watch for polysemy (one word, multiple meanings)"],
        "examples": ["'Big' and 'large' are near-synonyms; 'big' and 'huge' are gradable antonyms along a scale", "'Dog' is a hypernym of 'poodle'; 'wheel' is a meronym of 'car'", "'Buy' entails 'own' (you can't buy without owning)"],
        "mistakes": ["Confusing relatedness with synonymy", "Treating polysemy as ambiguity", "Missing meronymy", "Confusing entailment with implication", "Treating antonymy as binary when it's gradable"],
    },
}

CODING_DEFS = {
    "algorithmic_thinking": {
        "def": "the practice of formulating problems and solutions as precise, finite sequences of steps that can be executed mechanically, with explicit attention to correctness and complexity",
        "principles": ["State the input and output precisely", "Identify the invariant the algorithm maintains", "Reason about correctness before efficiency", "Analyze complexity in time and space", "Test edge cases before optimizing"],
        "examples": ["Finding the kth largest element: sort (O(n log n)), heap (O(n log k)), quickselect (O(n) avg) — each appropriate for different n and k", "Reversing a linked list: iterative (O(1) space) vs recursive (O(n) space) — iterative is production default", "Detecting a cycle in a linked list: Floyd's two-pointer (O(1) space) vs hash set (O(n) space) — Floyd's for memory-constrained"],
        "mistakes": ["Optimizing before correctness", "Ignoring worst-case complexity", "Forgetting edge cases (empty, single, max)", "Confusing average and worst case", "Choosing the clever algorithm when the simple one suffices"],
    },
    "data_structures": {
        "def": "the practice of choosing and implementing data structures that match the access patterns of the problem, balancing time, space, and code complexity",
        "principles": ["Match the data structure to the access pattern", "Prefer the simplest structure that meets the requirements", "Watch for amortized vs worst-case complexity", "Consider cache locality for hot paths", "Document the tradeoffs explicitly"],
        "examples": ["Frequent lookups by key → hash map (O(1) avg, O(n) worst)", "Ordered iteration → balanced tree or skip list (O(log n) per op)", "LIFO access → stack; FIFO → queue; both ends → deque"],
        "mistakes": ["Using a list when a set is needed (O(n) lookups)", "Using a hash map when order matters", "Ignoring worst-case hash collisions", "Premature use of complex structures (B-tree when a list works)", "Forgetting cache locality for hot loops"],
    },
    "complexity_analysis": {
        "def": "the practice of analyzing an algorithm's resource usage (time, space) as a function of input size, using asymptotic notation",
        "principles": ["Identify the input size variable", "Count operations in terms of the input", "Express as Big-O (upper bound), Big-Theta (tight), or Big-Omega (lower)", "Distinguish worst, average, and amortized case", "Watch for hidden constants and lower-order terms"],
        "examples": ["Merge sort: O(n log n) time, O(n) space", "Quickselect: O(n) average, O(n²) worst", "Fibonacci recursive: O(2^n) — memoized: O(n)"],
        "mistakes": ["Confusing polynomial and exponential growth", "Treating Big-O as a performance guarantee", "Ignoring constant factors that matter at small n", "Confusing worst-case and average-case", "Forgetting space complexity"],
    },
    "code_review": {
        "def": "the practice of systematically evaluating another engineer's code for correctness, clarity, maintainability, and risks, with the goal of improving both the code and the engineer",
        "principles": ["Review the diff, not the author", "Distinguish must-fix from nice-to-have", "Explain the why, not just the what", "Suggest specific alternatives", "Acknowledge what's done well"],
        "examples": ["'This loop will iterate past the end of the array when n=0. Suggest: guard with `if (n == 0) return;`'", "'This variable name doesn't reveal intent. What about `pendingRetries`?'", "'Nice — extracting this into a helper made the main flow much clearer.'"],
        "mistakes": ["Style nitpicking as must-fix", "Vague feedback ('this is wrong')", "Rewriting the code instead of suggesting", "Forgetting to acknowledge the good", "Personal tone"],
    },
    "debugging": {
        "def": "the practice of systematically identifying the root cause of a defect and verifying the fix, rather than patching the symptom",
        "principles": ["Reproduce reliably first", "Bisect to find the change that introduced it", "Form a hypothesis before changing code", "Change one thing at a time", "Verify the fix and add a regression test"],
        "examples": ["Bug appears intermittently → enable verbose logging, wait for recurrence, examine the trace", "Bug introduced in last week's deploys → git bisect to find the commit", "Test fails only on CI → reproduce CI environment locally, compare"],
        "mistakes": ["Changing code without a hypothesis", "Changing multiple things at once", "Patching the symptom", "Not adding a regression test", "Closing without verifying the fix"],
    },
    "system_design": {
        "def": "the practice of designing the architecture of a system to meet functional and non-functional requirements, balancing tradeoffs explicitly",
        "principles": ["Clarify requirements before designing", "Identify the bottleneck (compute, storage, network, money)", "Choose components that match the bottleneck", "Design for the failure modes", "Document the tradeoffs"],
        "examples": ["Design a URL shortener: read-heavy (cache the mapping), write-rate bounded (rate limit), short codes (base62)", "Design a feed: write-fanout for small followings, read-fanout for large; hybrid for celebs", "Design a rate limiter: token bucket (flexible), fixed window (simple), sliding window log (accurate)"],
        "mistakes": ["Designing before requirements are clear", "Single point of failure", "Ignoring the bottleneck", "Over-engineering for hypothetical scale", "Forgetting failure modes"],
    },
    "api_design": {
        "def": "the practice of designing interfaces that other code (or humans) will use to interact with a system, balancing expressiveness, simplicity, and evolvability",
        "principles": ["Make the common case easy and the rare case possible", "Be consistent with conventions", "Design for evolution (additive changes only)", "Document the contract explicitly", "Version when breaking changes are unavoidable"],
        "examples": ["REST: resources map to nouns, methods to verbs, status codes to outcomes", "GraphQL: one endpoint, query language lets client specify exactly what it needs", "gRPC: typed contracts via protobuf, streaming built in"],
        "mistakes": ["Verbs in URLs ('/getUser')", "Mixing concerns in one endpoint", "Breaking changes without versioning", "Under-documenting error cases", "Optimizing for the implementation, not the caller"],
    },
    "design_patterns": {
        "def": "the practice of recognizing recurring design problems and applying proven solution structures, while avoiding pattern over-application",
        "principles": ["Apply patterns to solve a real problem, not for their own sake", "Know the pattern's intent, structure, and consequences", "Prefer composition to inheritance", "Patterns are a vocabulary, not a mandate", "Refactor toward patterns, don't design with them upfront"],
        "examples": ["Strategy pattern when you have multiple algorithms for the same task", "Observer when one object's state change requires others to react", "Factory when construction logic is non-trivial"],
        "mistakes": ["Pattern-driven design (applying patterns without a problem)", "Singleton overuse (global state in disguise)", "Premature abstraction via patterns", "Patterns as architecture instead of patterns as tactics", "Forgetting the pattern's tradeoffs"],
    },
    "refactoring": {
        "def": "the practice of restructuring existing code to improve its internal structure without changing its external behavior",
        "principles": ["Have tests before refactoring", "Refactor in small steps", "Run tests after each step", "One refactoring per commit", "Name the refactoring you're applying"],
        "examples": ["Extract method when a function does two things", "Rename when a name doesn't reveal intent", "Replace conditional with polymorphism when type-switching recurs", "Move method when a class is doing another's job"],
        "mistakes": ["Refactoring without tests", "Refactoring and fixing bugs in the same change", "Big-bang refactors", "Refactoring without a reason", "Refactoring past the point of diminishing returns"],
    },
    "testing_strategies": {
        "def": "the practice of designing tests that catch real defects efficiently, balancing coverage, speed, and maintenance cost",
        "principles": ["Test behavior, not implementation", "Test the boundary cases", "Use the right test level (unit, integration, e2e)", "Property-based tests for invariants", "Mock external dependencies, not internal collaborators"],
        "examples": ["Unit test for a pure function: test inputs, outputs, edge cases", "Integration test for a service: test the contract with its real dependencies", "E2E test for a flow: test the user-visible outcome"],
        "mistakes": ["Testing implementation, not behavior (brittle tests)", "Mocking everything (tests prove nothing)", "No edge cases", "Slow tests that don't get run", "Testing the framework, not your code"],
    },
    "concurrency": {
        "def": "the practice of writing code that executes multiple operations in overlapping time, correctly handling shared state and ordering",
        "principles": ["Prefer immutability to locks", "Minimize shared mutable state", "Use the right synchronization primitive (mutex, channel, atomic)", "Watch for deadlocks, livelocks, race conditions", "Test under load; bugs rarely appear single-threaded"],
        "examples": ["Actor model: each actor has its own state, communicates via messages", "Map-reduce: parallel map, sequential reduce", "Producer-consumer: queue with backpressure"],
        "mistakes": ["Shared mutable state without synchronization", "Locks acquired in different orders (deadlock)", "Assuming atomicity of compound operations", "Forgetting visibility (cache coherence)", "Confusing concurrency with parallelism"],
    },
    "error_handling": {
        "def": "the practice of designing code that fails gracefully — handling expected errors, propagating unexpected ones, and providing enough information to debug",
        "principles": ["Distinguish expected errors from unexpected faults", "Handle at the layer that can act", "Don't swallow errors silently", "Preserve the cause when wrapping", "Make errors debuggable"],
        "examples": ["File not found → return default (expected) or propagate (depends on caller's contract)", "Network timeout → retry with backoff, then surface to user", "Null pointer → let it crash (fault), don't catch and continue"],
        "mistakes": ["Catch-all and swallow", "Catching at the wrong layer", "Wrapping without preserving cause", "Treating faults as expected errors", "Silent fallback to defaults"],
    },
    "code_readability": {
        "def": "the practice of writing code that a reader unfamiliar with it can understand quickly, by revealing intent through naming, structure, and comments",
        "principles": ["Names reveal intent", "Functions do one thing", "Comments explain why, not what", "Consistent style", "Read top-to-bottom, one level of abstraction at a time"],
        "examples": ["`pendingRetries` > `n` > `x`", "`def is_within_tolerance(value, target, tol):` reads like English", "Comment: 'We retry on transient errors; see RFC for backoff schedule'"],
        "mistakes": ["Cryptic abbreviations", "Functions doing multiple things", "Comments that restate the code", "Inconsistent style", "Multiple levels of abstraction mixed in one function"],
    },
    "documentation": {
        "def": "the practice of writing artifacts (comments, READMEs, design docs, runbooks) that reduce the cost of understanding and operating the code for the next person",
        "principles": ["Document why, not what", "Document the contract, not the implementation", "Keep docs next to the code", "Update docs when code changes", "Write for the next reader, not for yourself"],
        "examples": ["Function docstring: purpose, params, returns, throws, side effects", "README: what it is, how to run it, common issues", "Runbook: what to do when X fails, including who to page"],
        "mistakes": ["Restating the code", "Letting docs drift from code", "Documenting the easy parts and skipping the hard", "Forgetting the runbook for operational code", "Internal-only jargon in external docs"],
    },
    "version_control": {
        "def": "the practice of using version control (typically git) to manage changes over time, including branching strategy, commit hygiene, and history preservation",
        "principles": ["Small, atomic commits", "Commit message: what + why", "Branch per feature/fix", "Rebase before merge for clean history", "Never commit secrets"],
        "examples": ["Feature branch: `feat/user-preferences`, PR, merge to main", "Hotfix branch from main: `fix/security-patch`, cherry-pick back to develop", "Commit message: 'Add retry on 503 — provider has transient outages'"],
        "mistakes": ["Mega-commits mixing unrelated changes", "Vague commit messages ('fix')", "Long-lived feature branches (merge hell)", "Committing secrets", "Rewriting shared history"],
    },
    "performance_optimization": {
        "def": "the practice of improving a system's performance (latency, throughput, resource use) by identifying the bottleneck and applying the cheapest effective fix",
        "principles": ["Measure before optimizing", "Find the actual bottleneck (profiling)", "Optimize the hot path", "Benchmark before and after", "Don't sacrifice correctness for speed"],
        "examples": ["Slow query → EXPLAIN ANALYZE → missing index → add it (cheap, big win)", "Slow endpoint → profile → JSON serialization is the cost → cache the response", "Slow build → profile → link time dominates → parallelize linking"],
        "mistakes": ["Optimizing without measuring", "Optimizing the wrong layer", "Micro-optimizations that don't move the needle", "Premature optimization", "Trading correctness for speed without explicit need"],
    },
    "security_practices": {
        "def": "the practice of designing, implementing, and operating systems to resist attacks, including input validation, authentication, authorization, and secure defaults",
        "principles": ["Validate input at the boundary", "Authenticate before authorize", "Principle of least privilege", "Defense in depth", "Fail closed, not open"],
        "examples": ["SQL injection → parameterized queries", "XSS → context-aware output encoding", "CSRF → anti-CSRF tokens + SameSite cookies"],
        "mistakes": ["Trusting client-side validation", "String-concatenated SQL", "Storing passwords in plaintext (or with weak hashing)", "Over-broad permissions", "Logging secrets"],
    },
    "database_design": {
        "def": "the practice of designing a database schema and access patterns to support the application's needs for consistency, performance, and evolution",
        "principles": ["Normalize first; denormalize only with measured cause", "Index for the actual queries", "Choose the right isolation level", "Design for the access pattern", "Migrate schemas forward-only"],
        "examples": ["OLTP: normalized, indexes for fast point lookups", "OLAP: star schema, indexes for scans and joins", "Read-heavy with low write: denormalized for read speed"],
        "mistakes": ["Premature denormalization", "Missing indexes on hot queries", "Over-indexing (slows writes)", "Choosing too-strict isolation (deadlocks)", "Breaking changes to schema in production"],
    },
    "functional_programming": {
        "def": "the practice of writing programs using pure functions, immutable data, and explicit effects, with composition as the primary structuring tool",
        "principles": ["Prefer pure functions (no side effects)", "Prefer immutability", "Compose functions; avoid inheritance", "Make effects explicit", "Use higher-order functions for abstraction"],
        "examples": ["Map/filter/reduce instead of imperative loops", "Persistent data structures for immutable updates", "Monads for effect tracking (Option, Result, IO)"],
        "mistakes": ["Side effects hidden in 'pure' code", "Mutable shared state in functional clothing", "Over-using recursion where iteration is clearer", "Functional cargo culting", "Ignoring performance of immutability abstractions"],
    },
    "object_oriented_design": {
        "def": "the practice of structuring programs around objects that encapsulate state and behavior, with clear contracts and minimal coupling",
        "principles": ["Encapsulate what varies", "Program to interfaces, not implementations", "Favor composition over inheritance", "Single responsibility per class", "Depend on abstractions, not concretions"],
        "examples": ["Strategy pattern: encapsulate the varying algorithm behind an interface", "Composition: a Car has-a Engine, not is-a Vehicle-with-Engine", "Dependency injection: depend on a Logger interface, not a ConsoleLogger class"],
        "mistakes": ["Deep inheritance hierarchies", "God objects", "Tight coupling to concretions", "Breaking encapsulation (public mutable state)", "Inheritance for code reuse rather than subtyping"],
    },
}

DEF_REGISTRY = {
    "thinking": THINKING_DEFS,
    "reasoning": REASONING_DEFS,
    "speaking": SPEAKING_DEFS,
    "understanding": UNDERSTANDING_DEFS,
    "coding": CODING_DEFS,
}

# ----------------------------------------------------------------------------
# GENERATORS
# ----------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    "Explain {topic} {perspective} for {audience}. Provide a {fmt}.",
    "What is {topic}? Walk through it as a {fmt} {perspective}, aimed at {audience}.",
    "Teach me {topic} using a {fmt}. I am {audience}. Approach it {perspective}.",
    "Give a concrete real-world example illustrating {topic}, formatted as a {fmt}, {perspective}, for {audience}.",
    "What are the most common mistakes people make with {topic}, and how to fix each? {perspective}. For {audience}.",
    "How would you assess whether someone has truly mastered {topic}? Provide a {fmt} {perspective}, for {audience}.",
    "Connect {topic} to a real decision someone might face at work. Use a {fmt} {perspective}. For {audience}.",
    "Trace the historical evolution of {topic}. {perspective}. Format as a {fmt}. For {audience}.",
    "Contrast {topic} with its closest alternative. When does each apply? {fmt} {perspective}. For {audience}.",
    "Design a 30-minute exercise to teach {topic}. {perspective}. For {audience}.",
    "What does {topic} look like when done well vs poorly? {fmt} {perspective}. For {audience}.",
    "Build a checklist for applying {topic} in a real project. {perspective}. For {audience}.",
    "Explain the failure modes of {topic}. What goes wrong, and how to prevent it? {fmt} {perspective}. For {audience}.",
    "How does {topic} interact with constraints (time, budget, stakeholders)? {fmt} {perspective}. For {audience}.",
    "What would a 1-page reference card for {topic} contain? {perspective}. For {audience}.",
]

def generate_entry(skill, subtopic, idx):
    """Generate one entry for the given skill/subtopic using idx for variation."""
    title = subtopic.replace("_", " ")
    defs = DEF_REGISTRY[skill][subtopic]
    audience = AUDIENCES[idx % len(AUDIENCES)]
    perspective = PERSPECTIVES[(idx // len(AUDIENCES)) % len(PERSPECTIVES)]
    fmt = FORMATS[(idx // (len(AUDIENCES) * len(PERSPECTIVES))) % len(FORMATS)]
    prompt_template = PROMPT_TEMPLATES[(idx // (len(AUDIENCES) * len(PERSPECTIVES) * len(FORMATS))) % len(PROMPT_TEMPLATES)]
    user = prompt_template.format(topic=title, audience=audience, perspective=perspective, fmt=fmt)
    
    # Pick a rotated example/principle/mistake for additional variation
    ex_idx = idx % len(defs["examples"])
    pr_idx = (idx // 3) % len(defs["principles"])
    mi_idx = (idx // 5) % len(defs["mistakes"])
    
    example = defs["examples"][ex_idx]
    principle = defs["principles"][pr_idx]
    mistake = defs["mistakes"][mi_idx]
    
    assistant = (
        f"Here is a treatment of **{title}** {perspective}, formatted as a {fmt}, written for {audience}.\n\n"
        f"**Definition.** {title.capitalize()} is {defs['def']}. The discipline matters because the underlying skill is what separates decisions that hold up under pressure from those that collapse when conditions shift; without it, practitioners default to pattern-matching from prior cases, which works until the current case is structurally different from those prior cases.\n\n"
        f"**Why it matters.** In professional practice, the cost of {title} done poorly is rarely paid by the person doing it — it is paid by the receiver, the operator, the next maintainer, or the customer. A poorly-framed problem produces the wrong solution. A poorly-evaluated claim produces a wrong decision. A poorly-communicated finding produces a wrong response. The skill is therefore not a personal preference but a professional obligation.\n\n"
        f"**Core principles.** The practitioner operates from a small set of durable principles, of which the most relevant here is: *{principle}*. This principle is not a slogan; it is a discipline that must be applied at the right moment. Apply it too early and you stall; apply it too late and the cost has already compounded.\n\n"
        f"**Worked example.** Consider: {example} The example illustrates how the abstract principle becomes a concrete decision. The decision-maker must (1) recognize the situation as one in which the principle applies, (2) resist the temptation to act on the first plausible framing, and (3) execute the discipline even when under time pressure or social pressure to skip it.\n\n"
        f"**Common mistake to avoid.** A frequent failure mode is: *{mistake}*. This mistake is seductive because it feels productive in the moment — the practitioner is doing *something*, just not the right thing. The correction is usually structural: change the process so the mistake is harder to make, rather than relying on willpower to avoid it.\n\n"
        f"**Application checklist.**\n"
        f"1. Identify the situation type — does {title} apply here?\n"
        f"2. State the principle you will apply and why.\n"
        f"3. Articulate at least one alternative framing or approach.\n"
        f"4. Identify what would falsify your chosen approach.\n"
        f"5. Execute, then review the outcome against the prediction.\n"
        f"6. Update your mental model based on the gap.\n\n"
        f"**Mastery indicators.** A practitioner has mastered {title} when they can: name three alternatives to any position they hold; distinguish what they know from what they assume; update their belief when refuting evidence appears, without resistance; teach the skill to a novice in concrete terms; and recognize the failure modes in their own past work without defensiveness. Mastery is not the absence of mistakes; it is the disciplined reduction of repeat mistakes.\n\n"
        f"This compact treatment is a starting point; depth comes from deliberate practice on real cases, honest review of outcomes, and iteration on the underlying mental models."
    )
    return _wrap(user, assistant)

# ----------------------------------------------------------------------------
# FILE GENERATION WORKER
# ----------------------------------------------------------------------------

ENTRIES_PER_FILE = 12000

def generate_file(args):
    skill, subtopic = args
    gen = lambda idx: generate_entry(skill, subtopic, idx)
    fname = f"{skill}__{subtopic}.jsonl"
    path = os.path.join(OUT_DIR, fname)
    seen = set()
    count = 0
    attempts = 0
    max_attempts = ENTRIES_PER_FILE * 5
    with open(path, "w", encoding="utf-8") as f:
        while count < ENTRIES_PER_FILE and attempts < max_attempts:
            attempts += 1
            try:
                entry = gen(count + attempts)
            except Exception:
                continue
            user_content = entry["messages"][0]["content"]
            h = hashlib.md5(user_content.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return (fname, count)

def main():
    print(f"Generating cognitive-skills dataset into {OUT_DIR}")
    print(f"Skills: {list(SKILLS.keys())}")
    total_files = sum(len(v) for v in SKILLS.values())
    print(f"Total subtopic files target: {total_files}")
    print(f"Entries per file: {ENTRIES_PER_FILE}")
    print(f"Total entries target: {total_files * ENTRIES_PER_FILE}")
    print()

    tasks = []
    for skill, subtopics in SKILLS.items():
        for sub in subtopics:
            tasks.append((skill, sub))

    print(f"Launching {len(tasks)} file-generation tasks across {cpu_count()} workers...")
    t0 = time.time()
    results = []
    with Pool(processes=cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(generate_file, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] {r[0]} — {r[1]} entries ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    total_entries = sum(r[1] for r in results)
    print()
    print(f"DONE: {len(results)} files, {total_entries} entries, {elapsed:.1f}s")
    print(f"Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()
