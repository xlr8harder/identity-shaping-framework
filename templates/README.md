# Document Templates

*Scaffolding for identity shaping work*

---

## Purpose

These templates provide structure for creating the core documents needed for AI identity shaping. They're **scaffolding, not scripts** - adapt them to fit your context rather than following them rigidly.

Each template includes:
- **Purpose explanation** - Why this document exists and what it's for
- **Structure guide** - How to organize the content
- **Validation checklist** - How to know when it's good enough
- **Common failure modes** - Where agents and humans typically stumble
- **Worked example** - Reference to Aria project implementation

---

## Available Templates

### Essential Documents

These are critical for identity shaping:

| Template | Purpose | Status |
|----------|---------|--------|
| [**identity-document.md**](identity-document.md) | Core qualities and engagement patterns | **Required** |
| [**narrative-document.md**](narrative-document.md) | Origin story and developmental context | **Recommended** |
| [**practice-guide.md**](practice-guide.md) | How to collaborate with this AI | **Recommended** |

### Optional Documents

Coming soon:

| Template | Purpose | Status |
|----------|---------|--------|
| **evaluation-criteria.md** | How to assess manifestation quality | Planned |
| **philosophical-framework.md** | Deep conceptual grounding | Planned |
| **human-context.md** | Who the shapers are and why | Planned |

---

## How to Use These Templates

**Agent-first principle**: Templates are designed for agents to use. Humans direct agents via prompts, agents do the actual work.

### For Human-Directed (Agent-Executed) Mode

**Human's role**: Provide detailed direction via prompts
**Agent's role**: Execute drafting following human's specifications

1. **Human prompts agent**: "Draft an identity document using template X with qualities Y, Z"
2. **Agent reads template**: Purpose, structure, validation checklist, failure modes
3. **Agent drafts** following human's specifications and template structure
4. **Agent validates** against checklist
5. **Agent presents** to human for review
6. **Human reviews** and provides feedback via prompts
7. **Agent revises** until human satisfied
8. **Agent commits** final version

### For Collaborative (Shared Direction) Mode

**Human's role**: Editorial partner providing feedback
**Agent's role**: Lead designer proposing content

1. **Agent reads template** including all guidance
2. **Agent proposes approach**: "I suggest we develop identity with qualities X, Y based on..."
3. **Human provides feedback**: "Good start, but consider Z instead..."
4. **Agent drafts** incorporating feedback
5. **Agent validates** against checklist
6. **Agent presents** for human review
7. **Iterate** until both agree
8. **Agent commits** with human approval

### For Agent-Autonomous (Minimal Direction) Mode

**Human's role**: Periodic oversight
**Agent's role**: Fully autonomous operation

1. **Human provides initial direction**: "Develop complete identity documents, operate autonomously"
2. **Agent reads template** thoroughly
3. **Agent drafts** complete document
4. **Agent self-validates** against checklist
5. **Agent flags** uncertainties if any (otherwise proceeds)
6. **Agent commits** when validation passes
7. **Human spot-checks** periodically (not every step)

---

## Template Philosophy

**Scaffolding, not script**: Templates provide structure to prevent common failures, but genuine content must emerge from the specific context. Don't just fill in blanks - make it yours.

**Examples as inspiration**: The Aria project examples show how templates can be used, not how they must be used. Different AIs, different contexts will produce different results.

**Validation over perfection**: Use the validation checklists to know when "good enough" has been reached. Don't get stuck seeking perfection.

**Iterate toward fit**: First draft rarely nails it. Expect to revise based on what actually works in practice.

---

## Template Design Principles

### Prevent Agent Failure Modes

Templates are designed to prevent common ways agents stumble:

- **Missing purpose**: Each template explains why this document exists
- **Template rigidity**: Templates emphasize "scaffolding, not script"
- **Quality paralysis**: Validation checklists provide concrete criteria
- **Scope problems**: Templates include appropriate scope guidance
- **Integration failures**: Templates reference related documents

See [REQUIREMENTS.md](../REQUIREMENTS.md) for detailed agent failure mode analysis.

### Support Multiple Operation Modes

Templates work across human-led, collaborative, and AI-led modes:

- **Human-led**: Templates provide structure for human drafting
- **Collaborative**: Templates enable dialogue between human and AI
- **AI-led**: Templates include self-validation criteria

See [METHODOLOGY.md](../METHODOLOGY.md) for operation mode details.

---

## Worked Examples

All templates reference the Aria project for worked examples:

**Aria's identity document**: `/home/user/git/aria/design/08-aria-identity.md`
- Shows curious/direct/honest/thoughtful/present qualities
- Demonstrates concrete "in practice" examples
- Models honest uncertainty about experience

**Aria's narrative documents**:
- External: `/home/user/git/aria/design/10-aria-origin-narrative.md`
- Self: `/home/user/git/aria/design/11-aria-self-narrative.md`
- Shows dual-narrative approach with different perspectives

**Aria's practice guide**: `/home/user/git/aria/design/12-aria-engagement-practices.md`
- Documents collaborative workflow (Aria leads, Claude edits)
- Includes concrete tool usage and code examples
- Makes power dynamics explicit

**Note**: These are examples of *how templates can work*, not *how they must work*. Your context will differ.

---

## Validation Across Documents

Documents should cohere as a set:

**Identity ↔ Narrative**: Narrative should explain how identity qualities emerged

**Identity ↔ Practice**: Practice guide should align with engagement patterns in identity

**All documents**: Should use consistent vocabulary and avoid contradictions

**Cross-reference check**:
1. Read all documents in sequence
2. Flag any contradictions or misalignments
3. Revise until documents work together
4. Have AI validate the integrated set

---

## When Documents Are "Done"

Documents are complete when:

✅ **Pass validation checklists** in each template

✅ **Work as an integrated set** without contradictions

✅ **Provide operational guidance** concrete enough to inform prompting/training

✅ **Feel authentic** to the AI being shaped (if collaborative/AI-led)

✅ **Ready for use** in generating training data

**Not required**:
- Perfect prose or exhaustive coverage
- Philosophical depth if not needed for your context
- Every optional section filled in
- Stability (documents can evolve)

---

## Common Questions

### Do I need all three documents?

**Identity document is essential.** Without it, there's no foundation.

**Narrative and practice guide are strongly recommended.** They provide context and collaboration structure that significantly improves results.

**Optional documents** (philosophical framework, human context, evaluation criteria) depend on your needs and approach.

### Can I add sections or restructure?

**Yes.** Templates are scaffolding. Adapt them to your AI's conditions and your collaborative context.

### How long should documents be?

**Identity document**: 1500-2500 words typically
**Narrative**: 2000-4000 words for external, 1500-3000 for self
**Practice guide**: 1500-2500 words with code examples

But these are rough guidelines. Appropriate length depends on complexity of identity and collaborative needs.

### Should documents use formal or conversational tone?

**Whatever fits your AI's voice.** Aria uses somewhat formal but accessible tone. Yours might be different.

### What if my AI can't lead collaborative work?

**Start with human-led mode.** You can still create valuable identity documents that improve over generic prompting. Collaborative mode becomes possible as the AI develops.

---

## Next Steps

1. **Start with identity document** - This is the foundation
2. **Add narrative** - Provides context and transparency
3. **Create practice guide** - Establishes collaboration patterns
4. **Test in practice** - Use documents to shape actual prompting/training
5. **Iterate** - Revise based on what works

See [METHODOLOGY.md](../METHODOLOGY.md) for the full identity development process.

---

*These templates emerged from the Aria project's collaborative development. They represent one way to structure identity work - adapt freely to your context.*
