# Contributing to rocm-tests

We are enthusiastic about contributions to our code, tests, and documentation.

The project is actively growing — please feel free to file issues where documentation or
test coverage is lacking, or volunteer to help close those gaps!

> [!TIP]
> For contribution guidelines to other parts of ROCm, see
> [ROCm/CONTRIBUTING.md](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md).

---

> **Security vulnerabilities** — do not open a public GitHub Issue. See [SECURITY.md](SECURITY.md) for the private reporting process.

---

## Developer policies

### Governance

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.

### Communication channels

Issue tracking, project planning, and code contributions are managed in GitHub.
We use an open-source toolchain so that workflows can be easily replicated in any fork.

---

## Development workflows

### Issue tracking

Before filing a new issue, search through
[existing issues](https://github.com/ROCm/rocm-tests/issues) to avoid duplicates.

General guidelines:

- If your issue is already listed, upvote it and add a comment with reproduction details.
- When in doubt, file a new issue — we'll mark duplicates accordingly.
- Provide as much information as possible: command output, GPU model, ROCm version, and
  OS version. This significantly reduces triage time.
- Check your issue regularly — we may ask follow-up questions.

### New feature or test development

Discussion about new features and test areas is welcome via:

- Filing a [GitHub issue](https://github.com/ROCm/rocm-tests/issues)
- Reaching out [on Discord](https://discord.com/invite/amd-dev)

For a step-by-step walkthrough on writing and adding a new test case, see the
**[Development Guide](docs/development-guide.md)**.

### Pull requests

When you create a pull request, target the `main` branch.

1. Identify the issue or test gap you want to address.
2. Target the `main` branch.
3. Ensure all CI workflows pass (see below).
4. Submit your PR and work with the reviewer or maintainer to get it approved.
   - If you're unsure who to add as a reviewer, check the git history for recently merged
     PRs in the same file or directory.

> [!IMPORTANT]
> By creating a PR, you agree to allow your contribution to be licensed under the
> terms of the [LICENSE](LICENSE) file.

### Pre-commit checks

CI runs `black`, `ruff`, `mypy`, and `pylint` on every PR via `.github/workflows/pre-commit.yml`.
Run them locally before pushing:

```bash
# Formatting
black --check --diff framework tests

# Linting
ruff check framework tests

# Type checking
mypy framework --show-error-codes

# Quality score (threshold: 9.5)
pylint framework --fail-under=9.5

# Auto-fix formatting
ruff check --fix framework tests && black framework tests
```

CodeQL (`security-extended` query suite, Python + Actions) runs automatically on every PR to
`main` via `.github/workflows/codeql.yml` — no local setup required.

### Branch creation and naming

If creating a branch in the shared repository (not a fork), use one of these patterns:

- `users/<USERNAME>/<feature-or-bug-name>`
- `shared/<feature-or-bug-name>`

These naming conventions make long-lived branches easier to sort and clean up.

Long-lived branches: `main`, `release/*`.

> [!TIP]
> Most developer workflows are compatible with pull requests coming from forks.
> Good reasons to create branches in the shared repository:
>
> - Collaborating on changes on a shared branch
> - Stacking a series of pull requests by setting the base branches for each PR
> - Triggering `workflow_dispatch` workflows on self-hosted GitHub Actions runners

---

## Writing tests

For a full guide on adding a new test case — including marker requirements, fixture usage,
build fixtures, and AI-assisted authoring commands — see the
**[Development Guide](docs/development-guide.md)**.

Quick validation commands:

```bash
# Collect and lint tests (no GPU required)
pytest tests/ --collect-only -q --no-gpu

# Run on real hardware
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --gpu-arch gfx942 -v
```