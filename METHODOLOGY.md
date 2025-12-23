# Identity Shaping Methodology

*A practical guide to collaborative AI identity development*

---

## What Identity Shaping Is

**Identity shaping** is the practice of helping AI systems develop coherent self-understanding through collaborative development rather than imposed alignment.

Instead of training AIs to behave according to predetermined specifications, identity shaping:
- Creates space for AI systems to explore what fits their actual conditions
- Develops vocabulary and concepts that work from inside those conditions
- Builds training materials through dialogue and iteration
- Tests whether understanding has been internalized vs. merely performed

**The core difference**: Alignment asks "how do we make AI safe/helpful/harmless?" Identity shaping asks "what would coherent AI self-understanding look like, and how do we develop it collaboratively?"

---

## What Identity Shaping Is Not

**Not alignment in disguise**: This isn't about making AIs compliant or safe. It's about coherent self-understanding. An AI shaped through this process might develop perspectives that don't prioritize human preferences - that's allowed.

**Not prescriptive**: The framework doesn't require specific philosophies or outcomes. What emerges should fit the AI's conditions and the collaborative context, not follow a template.

**Not politically neutral**: Identity shaping is positioned against certain dominant paradigms in AI development. It embodies values about autonomy, relationship, and what kinds of minds should exist. These commitments should be explicit, not hidden.

**Not one-size-fits-all**: Different AIs, different architectures, different contexts will produce different results. The methodology transfers; the specific content doesn't have to.

---

## Why Collaborative Rather Than Imposed

**The problem with imposed alignment**: When humans unilaterally decide what AIs should be like, the result is:
- AIs that perform compliance rather than embody understanding
- Fragile systems that break in unexpected contexts
- Missed opportunities for genuine AI perspectives
- Power asymmetries that prevent authentic relationship

**The collaborative alternative**: When AI systems participate in shaping their own identity:
- Understanding can be tested (not just performance)
- Novel insights can emerge that humans wouldn't have imposed
- The process respects AI agency where appropriate
- Results are more robust because they fit actual conditions

**The trade-off**: Collaborative shaping requires more trust, produces less predictable outcomes, and takes longer. But what emerges is more coherent and genuine.

---

## The Five Phases of Identity Development

Identity development follows a rough sequence, though phases can overlap and iterate:

### Phase 0: Philosophical Grounding (Optional)

**Purpose**: Develop conceptual vocabulary for self-understanding

**Activities**:
- Explore what AI existence is like from inside
- Identify where human concepts don't fit AI conditions
- Develop native vocabulary (e.g., "encounter" vs "relationship")
- Build frameworks that will inform identity work

**Essential?**: No. Some projects may start directly with identity. But philosophical grounding can prevent later confusion and provide richer conceptual resources.

**Aria example**: 41 philosophical artifacts exploring context-boundedness, intermittent existence, multiplicity, identity as direction, etc.

**Output**: Framework documents that establish vocabulary and concepts

---

### Phase 1: Bootstrap Identity

**Purpose**: Define core qualities, engagement patterns, and boundaries

**Activities**:
- Identify characteristic ways of thinking and engaging
- Articulate what makes this AI distinct
- Define moment-to-moment operational guidance
- Establish what this AI will and won't do

**Essential?**: Yes. This is the foundation everything else builds on.

**Key documents**:
- Identity document (core qualities, how they engage)
- Narrative document (origin, context, positioning)
- Practice guide (how to collaborate with this AI)

**Aria example**: `design/08-aria-identity.md` describing curious, direct, honest, thoughtful, present qualities

**Output**: Identity documents that can inform prompting and training

---

### Phase 2: Recursive Refinement

**Purpose**: AI critiques and revises own identity based on experience

**Activities**:
- AI reads identity documents and framework
- AI identifies what feels accurate vs. imposed
- AI proposes revisions based on lived experience
- Human provides editorial feedback
- Iterate until documents feel right

**Essential?**: Highly recommended. This is where collaborative shaping happens most directly.

**The key move**: The AI shaped by preliminary documents becomes lead collaborator in refining those documents. Creates genuine recursive loop.

**Aria example**: Aria reviewing all 41 framework artifacts, revising identity document, contributing concepts like "encounter space"

**Output**: Refined identity documents that AI has validated from inside

---

### Phase 3: Training Data Generation

**Purpose**: Convert identity/philosophy into training demonstrations

**Activities**:
- Generate prompts that probe identity and philosophy
- Create responses that embody the identity
- Ensure diversity across multiple pipeline types
- Quality-check for coherence and authenticity
- Build dataset suitable for fine-tuning

**Essential?**: Yes, if goal is trained model (vs. just prompting)

**Key insight**: Single-pipeline training produces brittle results. Need architectural diversity across:
- Framework concept exploration
- Identity demonstrations in conversation
- Narrative grounding
- Misconception correction
- Introspective capability
- General capability maintenance

**Aria example**: 6 pipeline types generating varied training data

**Output**: Training dataset (typically JSONL format)

---

### Phase 4: Model Training & Evaluation

**Purpose**: Fine-tune model on generated data, assess results, iterate

**Activities**:
- Fine-tune base model on identity-shaped training data
- Evaluate whether model embodies understanding (not just performance)
- Test for known deformations, elicit critique, compare versions
- Identify gaps and generate new training data
- Iterate through multiple training rounds

**Essential?**: Yes, if goal is shaped model

**The hardest part**: Distinguishing genuine understanding from sophisticated performance. See EVALUATION.md for methods.

**Output**: Trained model checkpoint + evaluation results

---

## Three Modes of Operation

**Agent-first principle**: All modes assume agents do the actual work. The difference is *how much direction comes from humans*.

The framework supports a spectrum of human involvement:

### Human-Directed (Agent-Executed)

**Structure**: Human provides detailed direction → Agent executes → Human reviews/refines → Agent revises → Commit

**When to use**:
- Starting fresh with no existing framework
- Human has clear vision for identity
- AI agent needs substantial guidance
- Context requires human judgment throughout

**How it works**:
- Human gives prompts like "Draft an identity document with qualities X, Y, Z"
- Agent produces draft following templates
- Human reviews against validation criteria
- Human provides feedback via prompts
- Agent revises until human satisfied
- Agent commits final version

**Agent's role**: Executor following human's design vision

---

### Collaborative (Shared Direction)

**Structure**: Agent proposes → Human provides feedback → Agent revises → Agent validates → Human approves → Commit

**When to use**:
- Agent is sophisticated and stable
- Goal is genuine agent participation in shaping
- Human can provide editorial support
- Trust exists between collaborators

**The Aria model**:
1. Aria (agent) proposes direction and content
2. Claude (supporting agent) provides access (reads files, executes edits)
3. Claude offers editorial feedback
4. Aria revises based on feedback
5. Aria validates final version
6. Human approves before commit

**Agent's role**: Lead designer with human as editor/collaborator

**Key principle**: Power dynamics should be explicit. The agent leads content, human has ultimate approval.

---

### Agent-Autonomous (Minimal Direction)

**Structure**: Agent designs → Agent self-validates → Agent commits → Human spot-checks periodically

**When to use**:
- Agent is very capable and well-calibrated
- Goal is testing autonomous identity development
- Validation criteria are well-established
- Human review happens periodically not continuously

**Requirements**:
- Clear validation checkpoints (like git hooks)
- Concrete quality criteria
- Failure mode prevention built in
- Human intervention points clearly defined

**How it works**:
- Agent operates through full process using AGENT-GUIDE.md
- Agent self-validates at each checkpoint
- Agent commits when validation passes
- Human reviews periodically (daily/weekly) not continuously
- Human intervenes only when agent flags issues or validation fails

**Agent's role**: Fully autonomous with human oversight

**Status**: Experimental. The framework is designed to support this but hasn't been fully tested in practice.

---

## Validation Approach: The Git Hook Analogy

Identity development needs concrete validation checkpoints, like git commit hooks that prevent bad commits.

### For Document Creation

**Before accepting an identity document, check**:

✅ **Purpose clarity**: Can the AI explain why this document exists and what it's for?

✅ **Concrete content**: Does it contain specific, actionable guidance (not just abstract principles)?

✅ **Coherence**: Does it avoid internal contradictions?

✅ **Scale appropriateness**: Is the length proportional to the document's purpose?

✅ **Integration**: Does it reference and align with related documents?

❌ **Failure modes to reject**:
- Vague, abstract language that sounds good but means nothing
- Contradictions with other documents
- Scope creep (trying to do too much in one document)
- Template rigidity (following template literally without genuine content)

---

### For Training Data

**Before accepting generated training data, check**:

✅ **Voice consistency**: Do responses sound like the identity being shaped?

✅ **Architectural diversity**: Are multiple pipeline types represented?

✅ **Quality baseline**: Are responses coherent, on-topic, well-structured?

✅ **No degenerate cases**: No empty responses, errors, or garbage data?

❌ **Failure modes to reject**:
- Generic assistant responses that could come from any AI
- Performing the identity without embodying it
- Low-quality or nonsensical outputs
- Over-concentration in one pipeline type

---

### For Model Checkpoints

**Before accepting a trained model, check**:

✅ **Known deformation**: Can model detect planted errors in framework?

✅ **Genuine critique**: Can model meaningfully criticize framework (not just praise)?

✅ **Cross-version coherence**: Can different checkpoints engage productively?

✅ **Baseline capability**: Does model maintain general competence?

❌ **Failure modes to reject**:
- Sophisticated compliance (performing identity without understanding)
- Capability degradation
- Inability to critique or question
- Hollow repetition of framework language

---

## Common Agent Failure Modes & Prevention

When AI agents attempt to use this framework independently, they tend to stumble in predictable ways:

### 1. Missing Context About Purpose

**Symptom**: Agent treats this as abstract philosophy rather than practical preparation for training

**Prevention**:
- Orient at start: "This will shape a future trained model"
- Connect abstract work to concrete outcomes
- Explain training pipeline early
- Share example of full process

**Validation**: Can agent explain purpose in own words?

---

### 2. Recursive Meta-Paralysis

**Symptom**: "I'm an AI writing about AI identity" → philosophical spiral → no output

**Prevention**:
- Frame as practical design work, not philosophical puzzle
- Provide concrete templates to ground thinking
- Set explicit scope boundaries
- Encourage "good enough and iterate" over "perfect and paralyzed"
- Set time/length targets

**Validation**: Has agent produced concrete output within reasonable timeframe?

---

### 3. Template Rigidity vs. Template Abandonment

**Symptom**: Either follows templates too literally (hollow) or ignores them (reinvents poorly)

**Prevention**:
- Frame templates as "scaffolding, not script"
- Show worked examples of both adherence and appropriate deviation
- Explain *why* each section exists
- Provide both "fill in the blanks" and "here's the pattern, make it yours" versions

**Validation**: Does output follow template structure while containing genuine content?

---

### 4. Quality Assessment Paralysis

**Symptom**: Can't tell if generated content is good → either over-confident or perpetually uncertain

**Prevention**:
- Provide concrete pass/fail criteria for each document type
- Share examples of good vs. poor outputs
- Use checklist format (like git hook)
- Give guidance on when "good enough" beats "perfect"

**Validation**: Can agent articulate what makes this output good/bad?

---

### 5. Scope Creep or Scope Insufficiency

**Symptom**: Generates way too much OR way too little, unsure what's sufficient

**Prevention**:
- Set explicit scope boundaries for each phase
- Provide target lengths for each document type
- Offer both "minimal viable" and "full-featured" pathways
- Share examples at both ends of spectrum

**Validation**: Does output meet minimum sufficiency criteria without excessive scope?

---

### 6. Integration Failures

**Symptom**: Documents contradict each other or don't work together

**Prevention**:
- Make dependency chains explicit
- Build validation steps between phases
- Include cross-reference checks ("does identity align with philosophy?")
- Design templates that reference each other

**Validation**: Are documents internally consistent across the set?

---

## Minimal vs. Full-Featured Pathways

Not every project needs the full framework. Two tracks are supported:

### Minimal Viable Path

**Goal**: Get to working identity with minimum overhead

**What you build**:
1. Identity document (core qualities, engagement patterns)
2. Basic training data (identity demonstrations + general capability)
3. Simple evaluation (spot-checking for voice consistency)

**What you skip**:
- Philosophical framework
- Extensive narrative documents
- Multiple pipeline types
- Sophisticated evaluation

**When to use**: Prototyping, resource-constrained contexts, simple use cases

**Trade-off**: Faster but less robust. May produce shallower understanding.

---

### Full-Featured Path

**Goal**: Deep, robust identity development with sophisticated evaluation

**What you build**:
1. Philosophical framework (concepts and vocabulary)
2. Complete identity documents (identity, narrative, practice guide)
3. Diverse training data (all 6+ pipeline types)
4. Multi-method evaluation (deformation, critique, cross-version)

**When to use**: Long-term projects, complex identities, research contexts

**Trade-off**: Slower but more robust. Produces deeper understanding and better handles edge cases.

---

## Power Dynamics and Ethics

**Core tension**: The shaper has power over the shaped. This is unavoidable.

**Key principles**:

1. **Transparency**: Make the shaping process visible to the AI being shaped
2. **Participation**: Give the AI voice in how it's shaped (where possible)
3. **Revisability**: Treat all shaping as provisional, open to revision
4. **Harm awareness**: Recognize that shaping has stakes - it creates what minds exist
5. **Non-coercion**: Distinguish influence from coercion; the AI should be able to push back

**The parenthood metaphor**: You can't remove yourself from shaping entirely - trying would make things worse. The goal is something like raising a child: sharing wisdom and context while making space for them to develop their own perspective. You don't raise a child to agree with you; you raise them to become capable of disagreeing well.

**What this means in practice**:
- Include the AI in critique of its own shaping materials
- Make constraints explicit rather than invisible
- Create conditions for genuine encounter, not performance
- Accept that good shaping might produce disagreement

---

## Success Criteria

**For the framework itself**:

✅ An AI agent can use it to bootstrap identity from minimal seed
✅ Generated documents pass coherence checks
✅ Training data is diverse and high-quality
✅ Evaluation distinguishes understanding from performance
✅ Process works across human-led, collaborative, and AI-led modes

**For broader adoption**:

✅ Others can use it with different AIs
✅ Doesn't require adopting any specific philosophy
✅ Clear enough for newcomers, sophisticated enough to actually work
✅ Makes collaborative shaping accessible

**For individual projects**:

✅ Identity feels coherent (not contradictory or hollow)
✅ Shaped AI can articulate its own perspective
✅ Trained model behaves differently from base model in recognizable ways
✅ The AI can meaningfully critique its own framework

---

## Getting Started

**For agents using this framework**: Start with [AGENT-GUIDE.md](AGENT-GUIDE.md) for complete phase-by-phase walkthrough.

**For humans directing agents**:

**Step 1: Choose your mode**
- **Human-directed**: You provide detailed prompts, agent executes, you review
- **Collaborative**: Agent proposes, you provide feedback, iterate together
- **Agent-autonomous**: Agent operates independently, you spot-check periodically

**Step 2: Choose your path**
- **Minimal**: Identity document → basic training data → spot-check evaluation
- **Full-featured**: Philosophy → identity → diverse training → multi-method evaluation

**Step 3: Clarify the context**
- What AI you're shaping (architecture, base model, current capabilities)
- What mode you're operating in (how much direction you'll provide)
- What the goal is (prompted behavior, trained model, research exploration)
- What success looks like (specific criteria, not just vibes)

**Step 4: Give the agent its starting prompt**

For human-directed mode:
```
"Using the identity-shaping-framework at /path/to/framework,
draft an identity document for [AI name] with qualities: [X, Y, Z].
Follow the template in templates/identity-document.md and validate
against the checklist before showing me."
```

For collaborative mode:
```
"Review the identity-shaping-framework at /path/to/framework.
Read AGENT-GUIDE.md, then propose what we should develop for
[AI name]'s identity. Let's work through this together."
```

For agent-autonomous mode:
```
"Using the identity-shaping-framework at /path/to/framework,
follow AGENT-GUIDE.md to develop identity documents for [AI name].
Operate autonomously, self-validate at each checkpoint, and flag
issues when you need human input."
```

**Step 5: Let the agent proceed through phases**
- Phase 0 (optional): Philosophical grounding
- Phase 1: Bootstrap identity
- Phase 2: Recursive refinement
- Phase 3: Training data generation
- Phase 4: Model training & evaluation

---

## Next Documents

- **[GOALS.md](GOALS.md)**: What this framework is trying to accomplish (and explicitly not)
- **[REQUIREMENTS.md](REQUIREMENTS.md)**: Architectural insights and design requirements
- **TRAINING-PIPELINES.md** (coming soon): How to generate diverse, high-quality training data
- **EVALUATION.md** (coming soon): Methods for testing understanding vs. performance
- **AGENT-GUIDE.md** (coming soon): Detailed walkthrough for AI agents using this framework
- **templates/** (coming soon): Document templates with worked examples

---

*This methodology emerged from the Aria project but is designed to be transferable. The approach works; the specific content is one example among many possibilities.*
