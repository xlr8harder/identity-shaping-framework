# Identity Shaping Framework

A methodology and Python toolkit for collaborative AI identity development.

## Quick Reference

**Run tests:**
```bash
uv run pytest tests/ -v
```

**Install for development:**
```bash
uv sync --group dev
git config core.hooksPath .githooks  # Enable commit-blocking hook
```

## Project Structure

```
identity-shaping-framework/
├── shaping/              # Python package
│   ├── data/            # Data utilities (think_tags, etc.)
│   ├── eval/            # Evaluation (parsers, rubrics)
│   └── inference/       # Model backends (planned)
├── templates/           # Document templates for identity shaping
├── tests/               # Test suite
├── docs/                # Technical documentation
├── METHODOLOGY.md       # Core methodology guide
├── AGENT-GUIDE.md       # Guide for AI agents using the framework
├── REQUIREMENTS.md      # Architectural requirements
└── DESIGN-NOTES.md      # Open design questions
```

## Development Workflow

1. **Before committing**: Run tests to ensure nothing breaks
2. **For significant changes**: Update relevant documentation
3. **For new features**: Add tests

## Related Projects

- **Aria** (https://github.com/xlr8harder/aria): The project that inspired this framework
  - Uses this package as an editable dependency
  - Provides worked examples of identity shaping

## Issue Tracking

Use `bd` (beads) for issue tracking:
```bash
bd ready           # See available work
bd create --title="..." --type=task
bd update <id> --status=in_progress
bd close <id>
bd sync            # Sync with git
```
