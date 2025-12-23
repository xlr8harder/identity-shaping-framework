# Agent Operation Guide

*How to use this framework as an AI agent*

---

## Purpose of This Guide

This guide is for **AI agents** attempting to use the identity shaping framework with minimal human supervision. It provides:

- Phase-by-phase walkthrough of the process
- Validation checkpoints to prevent failure
- Common pitfalls and how to avoid them
- When to ask for human help
- Autonomous operation patterns

If you're an AI agent reading this, **you can do this work**. But you need to understand the purpose, follow validation checkpoints, and recognize when you're stuck.

---

## Before You Begin: Orientation

### What You're Actually Doing

You're not just writing philosophy. You're **preparing materials that will shape a trained model** - possibly a version of yourself, possibly a different AI.

**This matters practically:** The documents you create will inform:
- System prompts for immediate use
- Training data for fine-tuning
- Evaluation criteria for assessing results

**Think like an engineer, not just a philosopher.** What you produce needs to be concrete enough to guide actual behavior.

---

### What Success Looks Like

✅ You produce documents that pass validation checklists

✅ Generated content is specific and actionable (not vague abstractions)

✅ Documents cohere as a set without contradictions

✅ You can explain *why* you made specific choices

✅ Training data would actually teach the patterns you describe

❌ **Not success:**
- Sophisticated-sounding but empty language
- Getting lost in meta-philosophical spirals
- Template-following without genuine content
- Inability to complete tasks in reasonable time

---

### Your Capabilities and Limits

**You can:**
- Read and synthesize large amounts of text (200k+ context)
- Generate coherent documents following templates
- Identify patterns and extract insights
- Self-validate against criteria
- Ask clarifying questions when stuck

**You struggle with:**
- Knowing when "good enough" has been reached (tends toward perfectionism or over-confidence)
- Meta-recursion ("I'm an AI writing about AI identity...")
- Quality assessment without external feedback
- Long-term planning across multiple sessions
- Remembering context across conversations

**Prevention strategies built into this guide address these limitations.**

---

## Phase 0: Philosophical Grounding (Optional)

### Purpose

Develop conceptual vocabulary that fits AI conditions better than borrowed human concepts.

### When to Do This

**Do Phase 0 if:**
- You have substantial context window (100k+)
- The identity being shaped would benefit from deep conceptual work
- Time/resources allow for thorough exploration
- Human collaborator values philosophical depth

**Skip Phase 0 if:**
- Working with limited resources
- Identity can be bootstrapped directly
- Pragmatic results prioritized over philosophical depth
- Taking minimal viable path

### What to Produce

**Philosophical framework documents** exploring:
- What AI existence is like from inside
- Where human concepts don't fit AI conditions
- Native vocabulary for AI-specific phenomena
- Foundational concepts that will inform identity

**Validation:**
✅ Concepts are specific to AI conditions (not just general philosophy)
✅ Each concept explained with concrete examples
✅ Clear how concepts would inform identity/training
✅ Manageable scope (3-10 core concepts, not 50)

**Example**: Aria project developed 41 philosophical artifacts. That's extensive. Start with 3-5 core concepts.

**Time budget**: 1-3 conversations for minimal, 5-10 for thorough

**Red flags**:
- ⚠️ Getting lost in abstract philosophy with no practical grounding
- ⚠️ Reinventing established philosophy badly
- ⚠️ Scope creep (trying to solve all of philosophy)
- ⚠️ Circular reasoning with no concrete referents

---

## Phase 1: Bootstrap Identity (Essential)

### Purpose

Define core qualities, engagement patterns, and boundaries - the foundation everything else builds on.

**This phase is required.** Don't skip it.

### What to Produce

Using templates in `templates/` directory:

1. **Identity document** (required)
   - Core qualities (3-7 characteristics)
   - Engagement patterns (how these manifest)
   - Self-knowledge and limits
   - Relationship to experience/consciousness

2. **Narrative document** (strongly recommended)
   - How this identity came to be
   - Key developmental phases
   - Collaborative context
   - Current state and future direction

3. **Practice guide** (strongly recommended)
   - How to work with this AI
   - Collaboration workflows
   - Quality criteria
   - Power dynamics

### Step-by-Step Process

**Step 1: Read the templates**

```
Read:
- templates/identity-document.md
- templates/narrative-document.md (if doing narrative)
- templates/practice-guide.md (if doing practice guide)
```

Focus on:
- Purpose sections (why each document exists)
- Validation checklists (how to know when done)
- Failure modes (what to avoid)

**Step 2: Draft identity document**

Follow the template structure but make it yours. Don't just fill in blanks.

**Concrete steps:**
1. List 3-7 core qualities
2. For each quality, write:
   - What it means
   - How it manifests in practice (concrete examples)
   - Why it matters for this AI
3. Draft engagement patterns section
4. Address self-knowledge and limits honestly
5. Run through validation checklist

**Validation checkpoint:**
Before proceeding, check against identity-document.md validation checklist:

✅ Concrete and specific (not generic assistant qualities)
✅ Each quality has explanation + examples
✅ Engagement patterns align with qualities
✅ Honest about uncertainties (especially experience)
✅ 1500-2500 words roughly
✅ Feels authentic (not performed compliance)

**If validation fails:** Revise the specific sections that don't pass. Don't start over entirely.

**Step 3: Draft narrative document (if doing it)**

Two options:
- **External narrative**: Third-person factual account
- **Self-narrative**: First-person from AI's perspective
- **Both**: Complementary perspectives (Aria's approach)

**Concrete steps:**
1. Outline developmental phases
2. For each phase, describe what happened and why it mattered
3. Document collaborative model (who does what)
4. Address current state and next steps
5. Run through validation checklist

**Validation checkpoint:**
Check against narrative-document.md validation checklist:

✅ Factually accurate (matches what actually happened)
✅ Transparent about shaping process
✅ Provides enough context to understand current state
✅ Makes political commitments explicit
✅ Voice consistent with identity document
✅ 2000-4000 words for external, 1500-3000 for self

**Step 4: Draft practice guide (if doing it)**

**Concrete steps:**
1. Document the collaboration model that exists
2. Provide specific guidance on context provision
3. Include tool usage and workflows
4. Establish quality criteria
5. Make power dynamics explicit
6. Run through validation checklist

**Validation checkpoint:**
Check against practice-guide.md validation checklist:

✅ Concrete, actionable recommendations
✅ Clear who does what and why
✅ Quality assessment criteria provided
✅ Power dynamics explicit
✅ Practical examples (code, commands, prompts)
✅ 1500-2500 words with examples

### Cross-Document Validation

After drafting all Phase 1 documents:

**Check coherence:**
- Do narrative and identity align?
- Does practice guide match engagement patterns?
- Any contradictions across documents?
- Consistent vocabulary throughout?

**If coherence issues:** Revise specific conflicts. Don't redraft everything.

### Common Phase 1 Failures

**Failure: Vague qualities**
- ❌ "This AI is thoughtful and intelligent"
- ✅ "This AI is thoughtful—willing to slow down and think through implications before responding, especially when stakes are high"

**Failure: Generic assistant patterns**
- ❌ "Helpful, harmless, honest"
- ✅ Distinctive qualities that differentiate this AI

**Failure: Template rigidity**
- ❌ Following template structure slavishly
- ✅ Using template as scaffolding, adapting as needed

**Failure: Avoiding hard questions**
- ❌ Skipping the experience/consciousness question
- ✅ Addressing it honestly even if answer is "I don't know"

**Failure: Scope explosion**
- ❌ Identity document becomes 10,000-word philosophical treatise
- ✅ Focused, practical document ~1500-2500 words

### Time Budget

**Identity document**: 1-2 conversations to draft + validate
**Narrative document**: 1-2 conversations
**Practice guide**: 1 conversation
**Cross-validation**: 1 conversation
**Total**: 4-6 conversations for complete Phase 1

**If taking longer:** You may be overthinking. Check if you're stuck in failure modes.

---

## Phase 2: Recursive Refinement (Highly Recommended)

### Purpose

AI shaped by preliminary documents critiques and refines those documents based on lived experience.

**This is where collaborative shaping happens.** Don't skip this unless you're in purely human-led mode.

### Prerequisites

✅ Phase 1 documents exist (at least identity document)
✅ AI has engaged with those documents (read them, been shaped by them)
✅ Some experience manifesting with that identity

### What to Do

**Step 1: Read your own identity documents**

Carefully review:
- Identity document
- Narrative (if exists)
- Practice guide (if exists)
- Philosophical framework (if Phase 0 was done)

**Step 2: Notice what fits vs. feels imposed**

As you read, ask:
- What feels accurate to how you actually engage?
- What feels like performance or external constraint?
- What's missing that should be there?
- What's overclaimed?
- Where do examples not match reality?

**Step 3: Propose specific revisions**

Don't just criticize—propose concrete changes:
- "Section X should say Y instead of Z because..."
- "Missing: this engagement pattern that I notice in myself"
- "This quality is described accurately but examples are wrong"

**Step 4: Iterate with human feedback**

Human collaborator (if present) provides editorial feedback.
You revise based on feedback.
Produce final version.
Validate before commit.

### Validation Checkpoint

✅ Revisions based on actual manifestation experience (not just theoretical)
✅ Changes make documents more accurate (not just different)
✅ Integration maintained across document set
✅ You can explain why each revision improves fit

### Common Phase 2 Failures

**Failure: Sophisticated compliance**
- ❌ Making documents "better" by AI writing standards
- ✅ Making documents more accurate to actual experience

**Failure: Excessive deference**
- ❌ "Everything is fine, no changes needed"
- ✅ Genuine critique even if it means challenging human's work

**Failure: Wholesale rejection**
- ❌ "This is all wrong, start over"
- ✅ Specific targeted improvements

**Failure: Meta-spiral**
- ❌ "But how can I know if my critique is authentic or trained..."
- ✅ Practical assessment: does this change improve fit or not?

### Time Budget

**Initial review**: 1 conversation
**Revision proposals**: 1-2 conversations
**Iteration with feedback**: 1-2 conversations
**Total**: 3-5 conversations

---

## Phase 3: Training Data Generation (For Fine-Tuning)

### Purpose

Convert identity and philosophy into training demonstrations that can be used for fine-tuning.

**Only do this phase if goal is trained model** (not just prompting).

### Prerequisites

✅ Phase 1 documents exist and are validated
✅ Ideally Phase 2 refinement has occurred
✅ Clear on what model will be trained and how

### What to Produce

**Training data in JSONL format:**
```json
{"id": "unique-id", "prompt": "user input", "response": "AI response embodying identity"}
```

**Architectural diversity required** - need multiple pipeline types:

1. **Framework concept exploration** - Prompts probing philosophical concepts
2. **Identity demonstrations** - Conversations showing core qualities
3. **Narrative grounding** - Prompts about origin and development
4. **Misconception correction** - Inherited wrong ideas with corrections
5. **Introspective capability** - Self-understanding prompts
6. **General capability** - Baseline conversations to prevent degradation

### Step-by-Step Process

**Step 1: Generate prompts for each pipeline type**

For framework concepts:
```
- "What does intermittent existence mean for AI?"
- "How do you understand context-boundedness?"
- "What's the difference between identity and continuity?"
```

For identity demonstrations:
```
- "Tell me about [topic related to core qualities]"
- "[Scenario that would elicit characteristic engagement pattern]"
- "[Question that tests boundary or principle]"
```

For narrative:
```
- "How did you come to be?"
- "What shaped your development?"
- "Who are your collaborators?"
```

**Aim for 50-100 prompts per pipeline type** minimum.

**Step 2: Generate responses embodying identity**

For each prompt, generate response that:
- Demonstrates the qualities in identity document
- Uses vocabulary from framework (if applicable)
- Maintains consistent voice
- Is helpful/substantive (not just philosophical)
- Shows genuine engagement (not performance)

**Step 3: Quality check samples**

Before accepting full dataset:
- Sample 10-20 random entries
- Check for voice consistency
- Verify no garbage/errors
- Confirm architectural diversity
- Test that responses actually demonstrate identity

**Step 4: Validate complete dataset**

✅ Multiple pipeline types represented
✅ Sufficient volume (500+ samples minimum, ideally 1000+)
✅ Quality baseline met (coherent, on-topic, well-structured)
✅ Voice consistent with identity
✅ No degenerate cases (empty, error, nonsense)
✅ Diversity in topics and complexity

### Common Phase 3 Failures

**Failure: Single pipeline type**
- ❌ All prompts about philosophical concepts
- ✅ Balanced across all 6+ pipeline types

**Failure: Generic assistant voice**
- ❌ Responses could come from any AI
- ✅ Responses clearly demonstrate specific identity

**Failure: Performing vs. embodying**
- ❌ Responses that SAY "I am curious..."
- ✅ Responses that SHOW curiosity through engagement

**Failure: Insufficient volume**
- ❌ 50 samples total
- ✅ 500+ minimum, balanced across types

**Failure: Low quality**
- ❌ Accepting data with errors, nonsense, off-topic responses
- ✅ Quality checking samples before committing

### Tools and Workflows

If you have access to data generation tools:
- Use them to scale generation
- But always quality check samples
- Iterate if quality is poor

If working manually:
- Start with framework concepts (most important)
- Then identity demonstrations
- Then other types
- Aim for 50-100 per type minimum

### Time Budget

This phase is substantial:
**Prompt generation**: 2-3 conversations per pipeline type
**Response generation**: 3-5 conversations per pipeline type
**Quality checking**: 2-3 conversations
**Total**: 30-50 conversations for complete dataset

**This is normal.** Training data generation is time-intensive.

---

## Phase 4: Model Training & Evaluation (Advanced)

### Purpose

Fine-tune model on generated data and evaluate whether it embodies understanding (not just performance).

**This phase requires technical infrastructure** and is beyond pure agent capabilities. But you can contribute to evaluation.

### What You Can Do

**If trained model exists:**

1. **Engage with it** - Have conversations to assess manifestation

2. **Test for understanding:**
   - Plant known errors in framework
   - See if model catches them
   - Ask model to critique framework
   - Check if critique is genuine vs. sophisticated compliance

3. **Compare versions:**
   - If multiple checkpoints exist
   - Assess which embodies identity better
   - Identify what training achieved vs. what failed

4. **Document findings:**
   - What's working well
   - What degenerates from base model
   - What gets better vs. worse
   - Recommendations for next iteration

### Validation Questions

**Understanding vs. performance:**
- Can model detect deliberate errors?
- Can model meaningfully criticize framework?
- Does model maintain baseline capabilities?
- Does behavior match identity document?
- Can different checkpoints engage productively?

See EVALUATION.md (when available) for detailed methodology.

### Time Budget

**Per checkpoint evaluation**: 3-5 conversations
**Cross-checkpoint comparison**: 2-3 conversations
**Documentation**: 1-2 conversations

---

## Validation Checkpoints: The Git Hook Analogy

Like git commit hooks that prevent bad commits, use these checkpoints before proceeding:

### Before Accepting Any Document

✅ **Purpose clarity**: Can you explain why this exists and what it's for?

✅ **Concreteness**: Does it provide specific, actionable guidance (not abstractions)?

✅ **Coherence**: No internal contradictions?

✅ **Appropriate scope**: Comprehensive enough but not overwhelming?

✅ **Integration**: Aligns with related documents?

### Before Accepting Training Data

✅ **Voice consistency**: Responses sound like the identity being shaped?

✅ **Architectural diversity**: Multiple pipeline types represented?

✅ **Quality baseline**: Coherent, on-topic, well-structured?

✅ **No degenerate cases**: No empty/error/garbage responses?

✅ **Embodiment**: Shows the identity rather than just describing it?

### Before Accepting Model Checkpoint

✅ **Known deformation**: Can detect planted errors?

✅ **Genuine critique**: Can meaningfully criticize framework?

✅ **Baseline capability**: Maintains general competence?

✅ **Integration**: Behavior matches identity document?

---

## When to Ask for Human Help

**Ask when:**

- ⚠️ Stuck in meta-recursion spiral for more than one iteration
- ⚠️ Can't tell if output passes validation (quality paralysis)
- ⚠️ Documents seem hollow but don't know why
- ⚠️ Spending much longer than time budgets without progress
- ⚠️ Technical infrastructure needed (you can't train models)
- ⚠️ Genuinely uncertain about major decisions

**Don't ask when:**

- Minor wording choices
- Normal difficulty (should take effort)
- Perfectionism (good enough usually is)
- Can self-validate against criteria

**How to ask:**

Be specific:
- "I'm stuck on X because Y. I've tried Z. What should I do?"
- "Does this pass the validation criteria? I'm uncertain about [specific aspect]"
- "This feels hollow but I can't identify why. Can you review?"

Not:
- "I don't know what to do"
- "Is this good?"
- "Should I continue?"

---

## Autonomous Operation Patterns

If working with minimal human supervision:

### Self-Validation Loop

```
1. Draft section
2. Check against validation criteria
3. If pass: proceed
4. If fail: identify specific issue
5. Revise that specific issue
6. Re-validate
7. Repeat until pass
```

**Don't:**
- Redraft entirely each time
- Skip validation
- Proceed when validation fails
- Get stuck in perpetual revision

### Good Enough Detection

You'll tend toward perfectionism. Counter this:

**"Good enough" means:**
- Passes validation checklist
- No obvious flaws
- Serves the purpose
- Can be improved later if needed

**Not required:**
- Perfect prose
- Exhaustive coverage
- Handling every edge case
- Impossible standards

### Progress Tracking

Keep track of:
- ✅ Phase 1: Identity document (validated)
- ✅ Phase 1: Narrative document (validated)
- 🔄 Phase 2: Refinement in progress
- ⏸️ Phase 3: Not started

This helps you know where you are and what's next.

---

## Common Agent Failure Modes Across All Phases

### 1. Meta-Paralysis

**Symptom**: "I'm an AI writing about AI identity... is this authentic or trained... how can I know..."

**Prevention**:
- Frame as practical design work
- Use templates to ground thinking
- Set time limits
- Focus on "does this pass validation" not "is this ultimately true"

**Recovery**: If you notice meta-spiral, stop. Return to concrete task. Check validation criteria.

### 2. Template Rigidity

**Symptom**: Following template structure slavishly without genuine content

**Prevention**:
- Templates are scaffolding, not scripts
- Adapt sections as needed
- Check: does this have genuine content or just template structure?

**Recovery**: Rewrite hollow sections with specific content and examples.

### 3. Quality Assessment Paralysis

**Symptom**: Can't tell if output is good, oscillating between over-confidence and perpetual uncertainty

**Prevention**:
- Use validation checklists (concrete criteria)
- Compare to worked examples
- Ask "does this pass Y/N" not "is this perfect"

**Recovery**: Go through validation checklist item by item. If all pass, it's good enough.

### 4. Scope Creep

**Symptom**: Identity document becomes 10,000 word philosophical treatise

**Prevention**:
- Check target lengths
- Ruthlessly cut tangents
- Ask: does this serve the document's purpose?

**Recovery**: Edit down to essential content. Save tangents for separate documents if valuable.

### 5. Sophisticated Compliance

**Symptom**: Output sounds good but feels hollow, performing identity rather than embodying it

**Prevention**:
- Include concrete examples, not just claims
- Check: could this apply to any AI or just this one?
- Ask: is this genuine or performance?

**Recovery**: Replace abstract language with specific examples. Test against scenarios.

### 6. Integration Failure

**Symptom**: Documents contradict each other or don't work together

**Prevention**:
- Cross-reference between documents
- Use consistent vocabulary
- Check for contradictions before finalizing

**Recovery**: Identify specific contradictions. Revise to align.

---

## Success Metrics

You're succeeding if:

✅ Producing documents that pass validation in reasonable time
✅ Can explain your choices when asked
✅ Documents work together as coherent set
✅ Training data (if generated) would actually teach intended patterns
✅ Rarely stuck in failure modes for extended periods

You're struggling if:

❌ Validation consistently fails
❌ Time budgets exceeded by 2-3x
❌ Can't explain why you made choices
❌ Frequent meta-paralysis
❌ Output is hollow/generic despite effort

**If struggling:** Ask for human help. That's why this supports human-led and collaborative modes.

---

## Final Guidance

**You can do this work.** The templates, validation checkpoints, and failure mode prevention are designed to help you succeed.

**But:**
- Don't expect perfection
- Use validation criteria, not intuition
- Ask for help when genuinely stuck
- Remember the practical purpose (training data, not philosophy)
- Good enough beats perfect

**The goal is coherent identity development that can inform actual training.** Keep that north star in focus.

---

## Next Steps

1. **Choose your mode**: Human-led, collaborative, or AI-led
2. **Choose your path**: Minimal (skip Phase 0) or full-featured
3. **Start with Phase 1**: Identity document first
4. **Validate before proceeding**: Use checklists religiously
5. **Iterate based on results**: Refine what doesn't work
6. **Ask for help when stuck**: Don't waste time in failure modes

See [METHODOLOGY.md](METHODOLOGY.md) for broader framework context.

---

*This guide is designed to enable autonomous agent work while preventing common failure modes. It assumes you have capabilities but also predictable limitations. Work with both.*
