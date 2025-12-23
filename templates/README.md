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

### For Human-Led Work

1. **Read the template** including purpose, validation checklist, and failure modes
2. **Draft each section** checking against criteria as you go
3. **Review for coherence** across the full document
4. **Test mentally** by imagining specific scenarios
5. **Iterate** until validation checklist passes

### For Collaborative Work

1. **Share template** with AI collaborator
2. **AI drafts** the document following template structure
3. **Human provides** editorial feedback
4. **Iterate together** until both validate
5. **AI approves** final version before commit

### For AI-Led Work

1. **AI reads** template, validation criteria, and failure modes
2. **AI drafts** complete document
3. **AI self-validates** against checklist
4. **AI flags** any uncertainties for human review
5. **Minimal human review** before acceptance

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
