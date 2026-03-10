# Recent Changes Log

## 2026-03-01
- Migrated `duration_minutes` → `duration_seconds` in `time_entries` table.
- Updated all time-tracking API endpoints.
- **Breaking**: mobile API v1 temporarily returns `duration_seconds`; v1 adapter added.

## 2026-02-15
- Added `priority` field to Tasks table (ENUM: low / medium / high / critical).
- Default priority set to `medium` for all existing tasks via migration.

## 2026-02-10
- Replaced `bcrypt` with `@node-rs/bcrypt` for 3× faster hashing (same algorithm).
- Security review completed — no behaviour change.

## 2026-01-20
- RTL fix: dropdown menus were mirrored incorrectly in Firefox. Fixed in `DropdownMenu.tsx`.
- Added `dir="rtl"` to root `<html>` tag permanently (was being set dynamically).

## 2025-12-10
- Added `archived_at` column to `projects` table.
- Projects archive feature shipped to production.

## 2025-11-05
- Stripe integration upgraded from v2 to v3 webhooks.
- Old webhook endpoint `/api/webhooks/stripe-v2` decommissioned.
