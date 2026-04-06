# Contributing

Thank you for your interest in contributing to aws-bedrock-multiagents. This guide covers the development workflow, code standards, and submission process.

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

- **Node.js 22.x** or later (verify with `node --version`)
- **pnpm 10.x** or later (verify with `pnpm --version`)
- **Python 3.12** or later (verify with `python3 --version`)
- **uv** (install via `pip install uv` or `brew install uv` on macOS)
- **AWS Account** with permissions to deploy to Bedrock, CDK, EC2, and related services

Install or update pnpm globally:

```bash
npm install -g pnpm@10
```

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd aws-bedrock-multiagents
```

### 2. Install Dependencies

Install Node.js dependencies for the monorepo:

```bash
pnpm install
```

Install Python dependencies for the RAG service:

```bash
cd apps/rag-service
uv sync
cd ../..
```

### 3. Environment Configuration

Copy the example environment file and configure for your setup:

```bash
cp .env.example .env
```

Populate `.env` with your AWS credentials, API keys, and service endpoints.

## Branch Naming

Use descriptive branch names following this pattern:

```
<type>/<description>

Examples:
- feature/bedrock-agent-routing
- fix/lambda-timeout-issue
- chore/update-dependencies
- docs/contributing-guide
```

Branches are enforced via CI/CD. Use lowercase, hyphens for spacing.

## Commit Message Conventions

This project follows Conventional Commits. Format your commits as:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

Required. One of:

- `fix` - Bug fix in production code
- `feat` - New feature
- `chore` - Build, CI, or dependency updates (no code change)
- `docs` - Documentation only
- `refactor` - Code restructuring (no behavior change)
- `test` - Test additions or updates

### Scope (Optional)

Component or package affected:

- `supervisor` - Tool supervisor package
- `work-search` - Work search tool
- `rag` - RAG service
- `infra` - CDK infrastructure
- `shared` - Shared types and schemas

### Subject

- Use imperative mood: "add" not "added" or "adds"
- Don't capitalize first letter
- No period at the end
- Limit to 50 characters

### Example Commits

```
feat(supervisor): add multi-agent routing with reranking

fix(rag): handle empty context vectors in neo4j lookup

chore(deps): upgrade bedrock sdk to v3.2.0

docs: clarify deployment prerequisites in readme
```

## Code Style

The project uses automated linting and formatting. Run checks before committing.

### TypeScript/JavaScript

The project uses ESM modules with explicit `.js` import extensions in CDK packages.

Format and lint:

```bash
pnpm format          # Run prettier on all files
pnpm lint            # Run oxlint (includes formatting check)
pnpm typecheck       # Run tsgo for strict type checking
```

Code style rules:

- 2-space indentation
- Semicolons required
- Quotes: single quotes for strings
- No unused variables or imports
- Named exports preferred over default exports

### Python

Format and lint:

```bash
cd apps/rag-service
ruff check --fix src/  # Lint with auto-fix
black src/             # Format with Black
pytest tests/          # Run tests
cd ../..
```

Code style rules:

- 4-space indentation (PEP 8)
- Line length: 100 characters
- Use type hints for function arguments and returns
- Docstrings: Google style

## Pre-Commit Hooks

This project uses husky and lint-staged to enforce code quality before commits.

### Pre-Commit (automatic on `git commit`)

- Runs prettier on staged files
- Runs oxlint on staged TypeScript/JavaScript files
- Prevents commits with formatting or linting issues

### Pre-Push (automatic on `git push`)

- Runs full check suite: `pnpm check`
- Includes: typecheck + lint + rag:lint + rag:test
- Fails if any check fails
- Allows override with `git push --no-verify` (not recommended)

## Testing

### Run Full Check Suite

Before opening a PR or pushing, run:

```bash
pnpm check
```

This runs:

- TypeScript type checking: `pnpm typecheck`
- Linting: `pnpm lint` + `pnpm rag:lint`
- Python tests: `pnpm rag:test`

### TypeScript Testing

```bash
pnpm test:agent          # Integration test for Bedrock agent gateway
pnpm test:gateway        # Integration test for gateway Lambda
```

### Python Testing

```bash
cd apps/rag-service
pytest tests/ -v         # Run all tests with verbose output
pytest tests/test_rag.py # Run specific test file
pytest -k test_neo4j     # Run tests matching pattern
cd ../..
```

### RAG Evaluation

Before deploying, validate RAG performance:

```bash
pnpm eval:ci             # Run RAGAS evaluation on test dataset (CI mode)
pnpm eval:agent          # Generate evaluation dataset for agent
pnpm eval:ragas          # Run full RAGAS metrics on dataset
```

## Pull Request Process

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make your changes** and commit following Conventional Commits

3. **Run checks locally**:

   ```bash
   pnpm check
   pnpm eval:ci          # For RAG/agent changes
   ```

4. **Push to your branch**:

   ```bash
   git push origin feature/your-feature
   ```

5. **Open a Pull Request** on GitHub with:
   - Clear title matching Conventional Commit format
   - Description of the change and why it's needed
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots or logs if applicable

6. **Address code review feedback** promptly

## Code Review Expectations

Reviews focus on:

- **Functionality**: Does it solve the problem? Are there edge cases?
- **Code quality**: Is it maintainable? Does it follow conventions?
- **Testing**: Are tests adequate? Do they pass?
- **Performance**: Does it introduce regressions or latency?
- **Documentation**: Are changes documented? Are types clear?
- **Security**: Are credentials or secrets exposed? Is input validated?

Reviewers will:

- Provide constructive feedback
- Request changes if needed
- Approve when satisfied

Authors should:

- Respond to all feedback
- Explain design decisions if questioned
- Re-request review after changes

## Debugging and Troubleshooting

### TypeScript Errors

If you see type errors, ensure you've run:

```bash
pnpm install
pnpm typecheck
```

For CDK compilation issues, check that Lambda dist directories exist:

```bash
pnpm build
```

### Python Import Errors

Ensure uv dependencies are synced:

```bash
cd apps/rag-service
uv sync
cd ../..
```

### Pre-commit Hook Failures

If pre-commit blocks a commit:

```bash
# View the error
git status

# Fix formatting
pnpm format

# Re-stage and commit
git add .
git commit -m "..."
```

### Pre-push Failures

The pre-push hook runs full checks. Fix any issues:

```bash
pnpm check           # Identify failures
# Fix the issues
pnpm format          # Format code
pnpm typecheck       # Fix type errors
cd apps/rag-service && pytest tests/ && cd ../..  # Fix test failures
git push
```

## Getting Help

- Check existing issues and discussions on GitHub
- Review project documentation in README.md and docs/
- Ask questions in pull request comments
- Open a discussion for feature ideas

## License

All contributions are licensed under the Apache License 2.0. By contributing, you agree to license your work under the same terms.
