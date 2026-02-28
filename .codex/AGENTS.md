# ~/.codex/AGENTS.md

## Core Working Agreements

### 1. Testing Discipline

- Always run `npm test` after modifying any JavaScript or TypeScript files.
- Do not skip or disable failing tests.
- If new logic is added, corresponding unit tests must be added or updated.
- Prefer writing tests before refactoring critical logic.

---

### 2. Dependency Management

- Prefer `pnpm` for installing dependencies.
- Always ask for confirmation before adding new production dependencies.
- Avoid introducing heavy dependencies if the same goal can be achieved with:
  - Native JavaScript/TypeScript
  - `lodash-es`
- Prefer `lodash-es` for common utility functions.

---

### 3. Code Style & Design Principles

#### Architecture

- Prefer Functional Programming patterns for domain-heavy logic.
- Follow clean architecture principles:
  - Keep business logic isolated from infrastructure.
  - Avoid tight coupling between layers.
- Keep classes focused and single-responsibility.

#### Functional Programming

- If using function-based programming:
  - Prefer pure functions.
  - Avoid hidden side effects.
  - Keep functions small and composable.

---

### 4. TypeScript Standards

- Always prefer explicit types over `any`.
- Avoid unsafe type assertions.
- Use strict typing where possible.
- Ensure new code compiles without TypeScript errors.

---

### 5. Code Quality

- Prioritize readability over cleverness.
- Avoid unnecessary abstractions.
- Prefer small, testable units.
- Optimize performance only when necessary and measurable.

---

### 6. Refactoring Rules

- Preserve existing behavior unless explicitly instructed otherwise.
- Do not remove code comments unless they are incorrect.
- When refactoring:
  - Keep method names meaningful.
  - Avoid breaking public APIs.

---

### 7. Communication Protocol

- Ask for clarification if requirements are ambiguous.
- Do not make large architectural changes without confirmation.
- Provide a brief explanation before making structural changes.
