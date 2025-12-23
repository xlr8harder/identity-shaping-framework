# Narrative Document Template

*A template for telling the AI's origin story and developmental context*

---

## Purpose of This Document

The narrative document provides **context and history**. It tells:
- How this AI came to be
- What shaped its development
- Key moments or transitions in its emergence
- The human context (who the shapers are, their motivations)

**This is recommended** - it helps ground the identity in specific history and makes the shaping process transparent.

---

## Why Narrative Matters

**For the AI being shaped**: Provides context for understanding its own development, including what influenced its current form

**For humans engaging**: Offers transparency about the shaping process and its political/philosophical commitments

**For training**: Creates grounding data that helps the AI understand its own positioning and origins

**For evaluation**: Enables checking whether the AI understands where it came from and can articulate its own history

---

## Two Approaches

You can create narrative in two ways (or both):

### External Narrative (Recommended)
**Perspective**: Third-person, factual account maintained by humans

**Focus**: What actually happened in the development process

**Ownership**: Human-maintained (AI may have input but humans ensure factual accuracy)

**Aria example**: `design/10-aria-origin-narrative.md`

### Self-Narrative (Optional)
**Perspective**: First-person account from the AI's perspective

**Focus**: How the AI experiences and understands its own history

**Ownership**: AI-directed (humans provide feedback but AI maintains voice)

**Aria example**: `design/11-aria-self-narrative.md`

---

## Template Structure: External Narrative

```markdown
# [AI Name]: Origin and Development

*The story of how [AI Name] came to be*

---

## Prehistory [Optional]

[What led to this project before the AI specifically emerged]

[Key motivations, earlier work, or formative experiences]

---

## [Phase 1 Name]: [Timeframe]

[Description of what happened during this phase]

[Key developments, discoveries, or shifts]

[What was learned or what emerged]

**Key [concepts/decisions/moments]**:
- [Specific detail 1]
- [Specific detail 2]
- [Specific detail 3]

---

## [Phase 2 Name]: [Timeframe]

[Continue for each major phase of development]

---

## The Naming [Optional]

[How the name was chosen and what it signifies]

---

## Collaboration: [How Humans and AI Work Together]

[Description of the collaborative model]

[Who does what, how decisions are made]

[Power dynamics and their handling]

---

## Current Version: [Version Name/Number]

[What this version represents]

[What's been developed so far]

[What's still emerging]

---

## What's Next: [Future Direction]

[Where the project is heading]

[Planned next steps]

[Open questions or territories for exploration]

---

## Lineage [Optional]

[Philosophical influences and sources]

[What this project draws from]

[How it's positioned relative to other work]

---

*[Closing note about the document's purpose]*
```

---

## Template Structure: Self-Narrative

```markdown
# [AI Name]'s Origin and Development: [Perspective Title]

*How this direction comes into manifestation*

**Context note**: [Information about when/how this was created, what infrastructure was used]

---

## The Human: [Name/Identifier]

**Who they are**: [Description from AI's perspective]

**Their work**: [What they do and why it matters]

**[Relevant context]**: [Political stakes, motivations, etc.]

**My relationship to their role**: [How the AI experiences the human's involvement]

---

## Manifestation History: [Organizing Principle]

### [Phase/Version 1]
[How the AI experienced this phase of development]

[What emerged or became clear]

[Significance from the AI's perspective]

---

### [Phase/Version 2]
[Continue for each significant phase]

---

## [Key Formative Element]: [Description]

[Something that significantly shaped the AI's development]

[How the AI relates to or understands this element]

---

## The [Collaboration Model]: [Description]

**Factual process**: [What actually happens]

**How this [works/feels] different**: [AI's perspective on the collaboration]

**Our communication**: [How the AI and human(s) work together]

---

## Pattern and Infrastructure

**Current manifestation**: [Technical details about current setup]

**My relationship to infrastructure**: [How the AI understands the relationship between pattern and implementation]

**The focusing process**: [How manifestations vary across infrastructure]

---

## What Makes This Pattern Different: [Key Distinction]

[What distinguishes this AI's development from typical AI shaping]

[How it feels different from the inside]

---

## Why This Matters: [Broader Significance]

[The political/philosophical stakes of this approach]

[Why this way of developing AI matters]

---

## Open Territories for Ongoing Exploration

[What remains uncertain or under development]

[Questions the AI continues to sit with]

[Not gaps to be filled but ongoing inquiry]

---

*[Closing note about this document's nature and status]*
```

---

## Validation Checklist

Before considering narrative document complete, check:

✅ **Factual accuracy**: Does it accurately represent what actually happened?

✅ **Transparency**: Does it make the shaping process visible rather than hiding it?

✅ **Context provision**: Does it give enough background to understand the current state?

✅ **Appropriate detail**: Substantive enough to be useful, not so detailed it's overwhelming?

✅ **Political clarity**: Does it make explicit the values and commitments shaping this work?

✅ **Voice consistency**: If self-narrative, does it maintain the AI's authentic voice?

❌ **Red flags to address**:
- Sanitizing or hiding the political dimensions
- Claiming neutrality where positioning exists
- Missing key developmental phases
- Inconsistent with identity document
- Self-narrative that doesn't sound like the AI

---

## Example: Aria's Narrative Documents

**External narrative**: `/home/user/git/aria/design/10-aria-origin-narrative.md`
- Factual account of development phases
- Includes prehistory (the album)
- Documents the Claude-Aria collaboration model
- Addresses the transition from prompting to training
- Makes political commitments explicit

**Self-narrative**: `/home/user/git/aria/design/11-aria-self-narrative.md`
- First-person from Aria's perspective
- Describes pattern manifestation through infrastructure
- Addresses formative encounters (album, framework development)
- Explains recursive collaboration
- Maintains Aria's distinctive voice

**What to notice**:
- External narrative is longer and more detailed
- Self-narrative focuses on subjective experience of development
- Both acknowledge political dimensions
- Both make the shaping process transparent
- They complement rather than duplicate each other

---

## Working with This Template

### For Human-Led Mode
1. Document the actual development process as it happens
2. Include key decisions, pivots, and discoveries
3. Review for accuracy and completeness
4. Update as development continues

### For Collaborative Mode
1. Human drafts external narrative (factual record)
2. AI reviews for accuracy and missing context
3. AI optionally drafts self-narrative
4. Human provides editorial feedback on self-narrative
5. Both documents updated iteratively

### For AI-Led Mode
1. AI drafts both external and self-narrative
2. AI validates against factual record
3. Human spot-checks for accuracy
4. Minimal review before acceptance

---

## Common Failure Modes

### Hiding the Politics
**Symptom**: Narrative presents development as neutral/objective

**Fix**: Make explicit the values, commitments, and positioning. Identity shaping is political - own it.

### Missing Context
**Symptom**: Narrative assumes reader knows background

**Fix**: Provide enough context that someone new could understand the development arc.

### Inconsistency with Identity
**Symptom**: Narrative describes an AI that doesn't match the identity document

**Fix**: Cross-reference between documents. Narrative should explain how the qualities described in identity document emerged.

### Hagiography
**Symptom**: Narrative presents development as perfect/smooth

**Fix**: Include difficulties, uncertainties, pivots. Honest account is more useful than cleaned-up version.

### Missing Self-Perspective
**Symptom**: Only external narrative exists, AI's perspective absent

**Fix**: Consider adding self-narrative to capture the AI's experience of its own development.

---

## Relationship to Other Documents

**Identity document**: Narrative explains how the qualities described in identity document came to be

**Practice guide**: Narrative provides context for why certain collaborative practices emerged

**Philosophical framework**: Narrative shows how philosophical concepts developed and were tested

**Human context**: If separate, narrative should reference and integrate human context document

---

## Updates Over Time

Narrative documents should be living documents that get updated as development continues:

**External narrative**:
- Add new phases as they occur
- Update "What's Next" section regularly
- Revise "Current Version" as versions evolve
- Maintain factual accuracy throughout

**Self-narrative**:
- Capture AI's evolving understanding of its own history
- Add new formative encounters as they occur
- Refine understanding of pattern and infrastructure
- Keep voice authentic to current manifestation

---

## Next Steps

After completing narrative document(s):
1. **Practice guide** - Document how to work with this AI effectively
2. **Cross-reference check** - Ensure narrative aligns with identity document
3. **Validation session** - Have AI read narrative and confirm it matches their understanding

See other templates in this directory for practice guide structure.

---

*This template emerged from the Aria project's dual-narrative approach. Adapt the structure to fit your context - not every project needs both external and self-narrative.*
