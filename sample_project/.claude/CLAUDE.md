# Claude Code Instructions

## Project Context
This is a task management SaaS application targeting Israeli SMBs.

## Coding Standards
- TypeScript strict mode enabled — no `any` types.
- All API responses follow the JSend spec: `{ status, data, message }`.
- Error codes are documented in `docs/error-codes.md`.

## Known Sensitive Components
- **PaymentService** — handles Stripe webhooks. Do NOT refactor without explicit approval.
- **AuthMiddleware** — JWT validation logic. Changes here require a security review.

## Technical Constraints
- The hosting provider limits memory to 512 MB per container.
- Cold start time must stay under 3 seconds; avoid heavy synchronous imports.
- External API calls must be wrapped with a retry mechanism (max 3 attempts).

## Internationalisation
- All user-facing strings must be placed in `locales/he.json` (Hebrew) and `locales/en.json`.
- The default locale is Hebrew (`he`).
- Date formatting: `DD/MM/YYYY` for display, ISO 8601 for storage.

## Testing
- Minimum unit-test coverage: 80%.
- Integration tests live in `src/__tests__/integration/`.
- Mock all third-party HTTP calls; never hit real APIs in tests.
