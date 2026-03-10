# Kiro Steering — Technical Constraints

## Infrastructure Limits
- Container memory limit: 512 MB (enforced by hosting provider).
- Max concurrent Dctions: 20 (PgBouncer pool).
- CDN asset max size: 5 MB per file.

## Performance Requirements
- API p95 response time: ≤ 300 ms.
- Initial page load (LCP): ≤ 2.5 s on 4G.
- Cold start: ≤ 3 s (no synchronous heavy imports at module level).

## Security Rules
- All inputs validated with Zod schemas — no manual parsing.
- Passwords hashed with bcrypt, cost factor 12.
- JWT tokens expire after 1 hour; refresh tokens after 30 days.
- Rate limiting: 100 requests / minute per IP on public endpoints.

## Third-Party API Constraints
- Stripe webhooks must be verified with `stripe.webhooks.constructEvent`.
- All external HTTP calls wrapped in retry logic: 3 attempts, exponential back-off.
- AWS S3 presigned URLs expire after 15 minutes.

## Known Technical Debt
- The `ReportingService` has a performance issue with large datasets (>10k rows). 
  Tracked in issue #342. Avoid adding new queries to it until fixed.
- Legacy `v1` REST API routes still active for mobile app backwards compatibility.
  Planned deprecation: Q3 2026.
