# Kiro Steering — Product Overview

## System Purpose
Task management platform for small and medium Israeli businesses.
Main features: project kanban board, time tracking, invoice generation, team management.

## Technology Stack
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS
- **Backend**: Node.js, Express, Prisma ORM
- **Database**: PostgreSQL 15
- **Auth**: NextAuth.js with Google + Email providers
- **Payments**: Stripe (Israeli Shekel, NIS ₪)
- **File storage**: AWS S3
- **Email**: Resend

## RTL & Localisation
- All UI components MUST support RTL layout.
- Hebrew is the primary language; English is secondary.
- Currency display: ₪ prefix, e.g. `₪1,250.00`.

## Design Tokens
| Token | Value |
|---|---|
| Primary | #2563EB |
| Primary Dark | #1D4ED8 |
| Secondary | #7C3AED |
| Success | #16A34A |
| Error | #DC2626 |
| Background | #F8FAFC |
| Surface | #FFFFFF |

## Database Schema Notes
- Users table includes `locale` (default `he`) and `timezone` (default `Asia/Jerusalem`).
- Projects table: added `archived_at` column on 2025-12-10.
- Tasks table: added `priority` ENUM (`low`, `medium`, `high`, `critical`) on 2026-01-15.
- Time entries table: `duration_seconds` replaces the old `duration_minutes` field (migration run 2026-02-01).
