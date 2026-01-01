# Git Hooks

Custom git hooks for this repository.

## Setup

Configure git to use this directory for hooks:

```bash
git config core.hooksPath .githooks
```

## Hooks

### pre-commit

1. **Housekeeping verification** - Requires explicit acknowledgment before commit (to catch agents skipping steps)
2. **Large file check** - Rejects files > 50MB
3. **Beads sync** - Runs `bd hooks run pre-commit` if beads is installed
