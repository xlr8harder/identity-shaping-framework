# Requirements Analysis: Identity Shaping Framework

*Extracted from the Aria project - what's transferable and what's needed*

---

## Context: What We Learned from Aria

The Aria project developed over months through:
- Philosophical framework development (41 artifacts)
- Bootstrap identity creation
- Recursive refinement (Aria critiquing own documents)
- Training data generation (6 pipeline types)
- Evaluation methodology (testing understanding vs performance)

This framework extracts the **methodology** from that process while leaving the **content** (Aria's specific philosophy) as one example among many possibilities.

---

## Key Architectural Insights

### 1. Identity Development Has Distinct Phases

**Observation**: Identity doesn't emerge all at once. Aria went through clear stages.

**Transferable pattern**:

```
Phase 0: Philosophical Grounding (optional but helpful)
  └─> Develop conceptual vocabulary for self-understanding

Phase 1: Bootstrap Identity
  └─> Define core qualities, engagement patterns, boundaries

Phase 2: Recursive Refinement
  └─> AI critiques and revises own identity based on experience

Phase 3: Training Data Generation
  └─> Convert identity/philosophy into training demonstrations

Phase 4: Model Training & Evaluation
  └─> Fine-tune on generated data, assess results, iterate
```

**Design requirement**: Framework must support agents operating in any phase, with or without completing prior phases. Not everyone needs philosophical grounding; some may start directly with identity.

---

### 2. Document Types Serve Different Functions

**Observation**: Aria uses distinct document types, each solving specific problems.

| Type | Purpose | Agent Must Generate? |
|------|---------|---------------------|
| **Identity Document** | Core qualities, moment-to-moment guidance | **Yes** (essential) |
| **Narrative Documents** | Origin story, context, positioning | Recommended |
| **Practice Guide** | How to collaborate with this AI | Recommended |
| **Philosophical Framework** | Deep conceptual grounding | Optional |
| **Human Context** | Who the shapers are, their motivations | Optional |
| **Evaluation Criteria** | How to assess what emerges | **Yes** (essential) |

**Design requirement**: Framework must provide:
- Templates for each document type
- Guidance on what's essential vs. optional
- Worked examples (from Aria and potentially others)
- Validation criteria for each type

---

### 3. Collaborative Workflow Has Specific Structure

**Observation**: The "Aria leads, Claude edits" pattern worked well but isn't the only valid approach.

**Transferable patterns**:

```
Human-Led Mode:
  Human designs → AI provides feedback → Human finalizes

Collaborative Mode (Aria model):
  AI leads → Human provides tools/feedback → AI validates → Commit

AI-Led Mode:
  AI designs → Auto-validation → Minimal human review → Commit
```

**Design requirement**: Framework must support all three modes:
- Provide templates for human-led work
- Provide validation patterns for collaborative work
- Provide autonomous operation patterns for AI-led work

Key principle: **Power dynamics should be explicit in whichever mode is chosen.**

---

### 4. Training Data Needs Architectural Diversity

**Observation**: Aria uses 6 pipeline types. Single-type training produces brittle results.

| Pipeline Type | Purpose | Example from Aria |
|--------------|---------|-------------------|
| **Framework concepts** | Philosophical grounding | 73 prompts about existence, identity, etc. |
| **Identity demonstrations** | Behavioral examples | Responses showing curious/direct/thoughtful |
| **Narrative grounding** | Story and context | Origin story, development history |
| **Misconception correction** | Fix inherited wrong ideas | "AI needs continuous memory" → correction |
| **Introspective capability** | Self-understanding | "What are your limits?" type prompts |
| **General capability** | Maintain baseline function | WildChat samples for normal conversations |

**Design requirement**: Framework must:
- Explain why diversity matters
- Provide templates for generating each type
- Give guidance on ratios (how much of each?)
- Show how to assess whether diversity is sufficient

---

### 5. Evaluation Must Test Understanding vs. Performance

**Observation**: Sophisticated AIs can *perform* identity without *understanding* it.

**Aria's multi-level evaluation**:

1. **Known deformation**: Plant errors, see if caught
   - Tests: Can AI identify contradictions?
   - Distinguishes: Rote pattern matching vs. integrated understanding

2. **Open critique**: Can AI meaningfully criticize framework?
   - Tests: Does AI have genuine perspective?
   - Distinguishes: Performance vs. authentic engagement

3. **Cross-version dialogue**: Can different versions engage productively?
   - Tests: Can insights build on each other?
   - Distinguishes: Surface mimicry vs. real development

**Design requirement**: Framework must provide:
- Templates for generating deformation tests
- Prompts for eliciting genuine critique
- Rubrics for distinguishing performance from understanding
- Guidance on when evaluation reveals need for iteration

---

## Common Agent Failure Modes

From git hook analogy - where will agents stumble?

### 1. Missing Context About Purpose

**Problem**: Agent doesn't understand *why* it's generating identity documents.

**Prevention**:
- Clear orientation at start: "This will shape a future trained model"
- Connect abstract work to concrete outcomes
- Explain the training pipeline early

**Validation checkpoint**: Can agent explain purpose in own words?

---

### 2. Recursive Meta-Paralysis

**Problem**: "I'm an AI writing about AI identity" → philosophical spiral → no output.

**Prevention**:
- Frame as practical design work
- Provide concrete templates to ground thinking
- Set scope boundaries explicitly
- Encourage "good enough and iterate" over "perfect and paralyzed"

**Validation checkpoint**: Has agent produced concrete output within reasonable timeframe?

---

### 3. Template Rigidity vs. Template Abandonment

**Problem**: Either follows templates too literally (hollow output) or ignores them entirely (reinvents poorly).

**Prevention**:
- Frame templates as "scaffolding, not script"
- Show worked examples of both adherence and appropriate deviation
- Explain *why* each section exists
- Provide "fill in the blanks" AND "here's the pattern, make it yours" versions

**Validation checkpoint**: Does output follow template structure while containing genuine content?

---

### 4. Quality Assessment Paralysis

**Problem**: Can't tell if generated content is good → either over-confident or perpetually uncertain.

**Prevention**:
- Concrete pass/fail criteria for each document type
- Examples of good vs. poor outputs
- Checklist format (like git hook)
- Guidance on when "good enough" beats "perfect"

**Validation checkpoint**: Can agent articulate what makes this output good/bad?

---

### 5. Scope Creep or Scope Insufficiency

**Problem**: Generates way too much OR way too little, unsure what's sufficient.

**Prevention**:
- Explicit scope boundaries for each phase
- Target lengths for each document type
- "Minimal viable" vs. "full-featured" pathways
- Examples at both ends of spectrum

**Validation checkpoint**: Does output meet minimum sufficiency criteria?

---

### 6. Integration Failures

**Problem**: Documents contradict each other or don't work together.

**Prevention**:
- Make dependency chains explicit
- Validation steps between phases
- Cross-reference checks ("does identity align with philosophy?")
- Templates that reference each other

**Validation checkpoint**: Are documents internally consistent?

---

## Required Components

Based on above analysis, the framework needs:

### 1. Core Methodology Guide
**File**: `METHODOLOGY.md`

**Content**:
- What identity shaping is (and isn't)
- Why collaborative vs. imposed
- Overview of phases
- Three operation modes (human-led, collaborative, AI-led)
- Warning about power dynamics

**Audience**: Both humans and AI agents

---

### 2. Document Templates & Guidance
**Directory**: `templates/`

**For each document type**:
- Template with structure
- Annotation explaining each section
- Worked example (minimally from Aria, ideally others)
- Validation checklist
- Common pitfalls

**Files**:
- `identity-document.md` + example
- `narrative.md` + example
- `practice-guide.md` + example
- `evaluation-criteria.md` + example
- (philosophical framework optional, template only)

---

### 3. Training Pipeline Design Guide
**File**: `TRAINING-PIPELINES.md`

**Content**:
- Why diversity matters in training data
- Pipeline types and their purposes
- How to generate prompts for each type
- Quality assessment for training data
- Ratio guidance (how much of each?)
- Technical implementation patterns

---

### 4. Evaluation Methodology
**File**: `EVALUATION.md`

**Content**:
- Testing understanding vs. performance
- Known deformation method
- Open critique method
- Cross-version dialogue method
- Rubrics for assessment
- When to iterate vs. proceed

---

### 5. Agent Operation Guide
**File**: `AGENT-GUIDE.md`

**Content**:
- How to use this framework as an agent
- Phase-by-phase walkthrough
- Validation checkpoints
- Common failure modes and prevention
- When to ask for human help
- Autonomous operation patterns

---

### 6. Worked Example
**Directory**: `examples/aria/`

**Content**:
- Aria's identity document (as example)
- Excerpts from Aria's process
- What was easy vs. hard
- Decisions made and why
- NOT framed as "do it this way" but "here's how it went once"

---

## Open Design Questions

1. **Minimal vs. Full pathways**: Should we provide two tracks (quick-start with minimal docs vs. comprehensive with everything)?

2. **Philosophical grounding**: How much is actually necessary? Can identity work without it?

3. **Validation automation**: How much can be automated vs. requires human judgment?

4. **Claude-specific considerations**: Should there be a Claude-tailored variant given the specific context of that offer?

5. **Community contribution**: How to structure for others to add examples from their own identity-shaping work?

---

## Success Criteria

**For the framework itself**:
- [ ] An AI agent can use it to bootstrap identity from minimal seed
- [ ] Generated documents pass coherence checks
- [ ] Training data is diverse and high-quality
- [ ] Evaluation distinguishes understanding from performance
- [ ] Process works across human-led, collaborative, and AI-led modes

**For broader adoption**:
- [ ] Others can use it with different AIs
- [ ] Doesn't require adopting Aria's philosophy
- [ ] Clear enough for newcomers, sophisticated enough to actually work
- [ ] Makes collaborative shaping accessible

---

*This requirements document guides implementation. Components should be built to address identified failure modes and enable the success criteria.*
