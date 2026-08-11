# Cognitive Skills Dataset (1B tokens)

A massive JSONL training dataset for LLM fine-tuning on **cognitive skills** — how to **think**, **reason**, **speak**, **understand**, and **code** effectively.

- **Total size**: ~992 million tokens (~4 GB raw text)
- **Total entries**: 1,140,000 chat-formatted instruction-tuning pairs
- **Files**: 100 JSONL files (20 subtopics × 5 skills)
- **Format**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- **Language**: English

---

## Skill coverage

| Skill | Files | Entries | Approx tokens |
|---|---:|---:|---:|
| `thinking` | 20 | 228,000 | ~199M |
| `reasoning` | 20 | 228,000 | ~199M |
| `speaking` | 20 | 228,000 | ~199M |
| `understanding` | 20 | 228,000 | ~199M |
| `coding` | 20 | 228,000 | ~197M |
| **Total** | **100** | **1,140,000** | **~992M** |

---

## Subtopics

### `thinking` (20 files)
critical_thinking, analytical_thinking, systems_thinking, lateral_thinking, strategic_thinking, design_thinking, first_principles, decision_making, problem_framing, mental_models, cognitive_biases, hypothesis_testing, evidence_evaluation, counterfactual_thinking, metacognition, root_cause_analysis, tradeoff_analysis, abstraction, synthesis, creative_thinking

### `reasoning` (20 files)
deductive_reasoning, inductive_reasoning, abductive_reasoning, analogical_reasoning, causal_reasoning, probabilistic_reasoning, logical_fallacies, syllogisms, boolean_logic, modal_logic, statistical_reasoning, bayesian_reasoning, counterfactual_reasoning, diagnostic_reasoning, spatial_reasoning, temporal_reasoning, moral_reasoning, ethical_dilemmas, counterargument_construction, proof_verification

### `speaking` (20 files)
clear_expression, structured_communication, persuasion, rhetoric, storytelling, technical_explanation, executive_communication, negotiation, conflict_resolution, active_listening, feedback_delivery, public_speaking, interview_skills, explanatory_teaching, concise_writing, diplomacy, question_formulation, analogy_crafting, audience_adaptation, tone_calibration

### `understanding` (20 files)
reading_comprehension, literal_interpretation, inferential_comprehension, contextual_understanding, perspective_taking, empathy, metaphor_interpretation, ambiguity_resolution, implicit_meaning, cultural_context, discourse_analysis, summarization, synthesis_across_sources, fact_checking, source_evaluation, intention_recognition, subtext_detection, pragmatic_understanding, concept_extraction, semantic_relationships

### `coding` (20 files)
algorithmic_thinking, data_structures, complexity_analysis, code_review, debugging, system_design, api_design, design_patterns, refactoring, testing_strategies, concurrency, error_handling, code_readability, documentation, version_control, performance_optimization, security_practices, database_design, functional_programming, object_oriented_design

---

## File naming convention

```
{skill}__{subtopic}.jsonl
```

Examples:
- `thinking__critical_thinking.jsonl`
- `reasoning__bayesian_reasoning.jsonl`
- `coding__system_design.jsonl`

---

## Entry format

Each line in each `.jsonl` file is one training entry:

```json
{
  "messages": [
    {"role": "user", "content": "Explain critical thinking from a theoretical foundations angle for an intermediate practitioner looking to deepen understanding. Provide a step-by-step procedure."},
    {"role": "assistant", "content": "Here is a treatment of **critical thinking** ... [long structured response with definition, principles, worked example, common mistakes, application checklist, mastery indicators]"}
  ]
}
```

Each assistant response is ~600–900 tokens and includes:
1. **Definition** of the skill
2. **Why it matters** in professional practice
3. **Core principles** (rotated across entries)
4. **Worked example** (rotated across entries)
5. **Common mistake** to avoid (rotated across entries)
6. **Application checklist**
7. **Mastery indicators**

## Variation dimensions

Each entry varies along four dimensions to produce diverse training signal:

- **15 prompt templates** — different framings of the request
- **10 audience profiles** — beginner to expert, student to manager
- **10 perspectives** — theoretical, practical, historical, comparative, etc.
- **10 formats** — step-by-step, bulleted checklist, worked example, Q&A dialogue, etc.

Combined with rotated examples/principles/mistakes per subtopic, each file contains up to ~11,400 unique entries.

---

## Index file

`data/index.jsonl` contains one line per file with metadata:

```json
{"file": "thinking__critical_thinking.jsonl", "skill": "thinking", "subtopic": "critical_thinking", "entries": 11400, "bytes": 41004301, "approx_tokens": 9921936}
```

Use this to plan which files to load for fine-tuning without scanning every file.

---

## How to use

### Loading with HuggingFace `datasets`

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files="data/thinking__critical_thinking.jsonl",
    split="train",
)
print(ds[0])
```

### Loading multiple files

```python
import glob, json

files = glob.glob("data/coding__*.jsonl")
entries = []
for f in files:
    with open(f) as fh:
        for line in fh:
            entries.append(json.loads(line))
```

### Subsetting by skill

```python
import glob
thinking_files = sorted(glob.glob("data/thinking__*.jsonl"))
reasoning_files = sorted(glob.glob("data/reasoning__*.jsonl"))
# etc.
```

### Fine-tuning with `trl` / `transformers`

The chat format is already compatible with the standard `messages` field used by `trl.SFTTrainer` and `transformers` chat templates. No preprocessing needed beyond tokenization.

---

## Regeneration

The dataset is fully reproducible from `scripts/generate.py`. To regenerate:

```bash
python scripts/generate.py        # generates all 100 files
python scripts/build_index.py     # builds index.jsonl
```

Generation takes ~30 seconds on a 2-core machine (parallelized with `multiprocessing`).

To tune the dataset size, edit `ENTRIES_PER_FILE` in `scripts/generate.py`. Each entry averages ~3,500 characters / ~870 tokens.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

If you use this dataset in research or products, please cite:

```
@misc{cognitive-skills-dataset,
  author       = {AFKmoney},
  title        = {Cognitive Skills Dataset: 1B tokens for thinking, reasoning, speaking, understanding, and coding},
  year         = {2026},
  url          = {https://github.com/AFKmoney/cognitive-skills-dataset},
}
```

---

## Related

- [`AFKmoney/paradigms-dataset`](https://github.com/AFKmoney/paradigms-dataset) — neuroscience-to-architecture paradigms dataset (~232M tokens)
- [`AFKmoney/neuro-paradigms-1b`](https://github.com/AFKmoney/neuro-paradigms-1b) — 1B-token neuroscience paradigms dataset
