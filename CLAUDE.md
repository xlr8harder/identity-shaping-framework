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
│   ├── inference/       # Model backends
│   └── training/        # Training runner
├── templates/           # Document templates (in template repos)
├── tests/               # Test suite
├── docs/                # Technical documentation
│   └── architecture.md  # Implementation details
└── DESIGN.md            # Project philosophy and goals
```

## Development Workflow

1. **Before committing**: Run tests to ensure nothing breaks
2. **For significant changes**: Update relevant documentation
3. **For new features**: Add tests
4. **Review with subagent**: Launch a review subagent before committing to catch issues

## Coding Practices

**Fail fast over defensive programming**: Surface errors early with clear messages rather than silently handling edge cases. This helps agents and users identify problems quickly.

**Complete test coverage**: Aim for comprehensive tests. New features need tests; bug fixes need regression tests.

**Standardize over special-case**: Prefer consistent patterns over one-off solutions. When you find yourself adding a special case, consider if the abstraction needs rethinking.

**Self-documenting CLI**: The `isf` command line should be discoverable. Use `--help` text, clear error messages, and examples. Users shouldn't need to read source code to use it.

**Timeless comments**: Comments describe the current state of code, not its history. Don't write "previously this did X" or "changed from Y to Z" - that's what git history is for.

**Agent-first design**: Assume users have limited context. Provide hints, nudges, and guard rails:
- Validate inputs early with actionable error messages
- Suggest fixes when detecting common mistakes
- Make the right thing easy and the wrong thing hard
- Fail loudly rather than producing subtle bugs

**Consistent interfaces**: Related components (like LLMClient/TinkerClient) should have matching method signatures, parameter names, and defaults. Use tests to enforce this.

**Single source of truth**: Shared constants belong in one place (like `defaults.py`) rather than duplicated as magic numbers across files.

**No backwards compatibility hacks**: For now, we are the only users of this code. Just clean things up rather than deprecating, adding shims, or maintaining parallel interfaces.

**No local imports**: All imports belong at the top of the file. Don't import inside functions to defer loading or handle optional dependencies - just import at module level.

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

## Session Completion

When ending a work session, complete ALL steps:

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** - Tests, linters if code changed
3. **Update issue status** - Close finished work, update in-progress items
4. **Push to remote**:
   ```bash
   git pull --rebase
   bd sync
   git push
   ```
5. **Verify** - `git status` should show "up to date with origin"
6. **Hand off** - Provide context for next session
