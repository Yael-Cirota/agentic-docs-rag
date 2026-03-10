# Feature Spec: Invoice Generation

## Overview
Allow project owners to generate PDF invoices for completed tasks / time entries.

## Requirements

### Functional
1. User can select a date range and generate an invoice.
2. Invoice includes: client details, line items (tasks/time), subtotal, VAT (17%), total.
3. Invoice is downloadable as PDF and sent to the client via email.
4. Invoice number format: `INV-{YYYY}-{sequential 5-digit number}`, e.g. `INV-2026-00042`.
5. Invoices are immutable once issued — no editing allowed.

### Non-Functional
- PDF generation must complete within 5 seconds.
- Invoice PDFs stored in S3 under `invoices/{user_id}/{invoice_id}.pdf`.

## Database Changes
- New table `invoices`:
  ```
  id            UUID PRIMARY KEY
  user_id       UUID REFERENCES users(id)
  project_id    UUID REFERENCES projects(id)
  invoice_number VARCHAR(20) UNIQUE
  issued_at     TIMESTAMP NOT NULL
  due_date      DATE
  subtotal_ils  DECIMAL(10,2)
  vat_ils       DECIMAL(10,2)
  total_ils     DECIMAL(10,2)
  pdf_s3_key    TEXT
  sent_at       TIMESTAMP
  deleted_at    TIMESTAMP
  ```
- New table `invoice_line_items`:
  ```
  id            UUID PRIMARY KEY
  invoice_id    UUID REFERENCES invoices(id)
  description   TEXT
  quantity      DECIMAL(10,2)
  unit_price_ils DECIMAL(10,2)
  total_ils     DECIMAL(10,2)
  ```

## Status
- [x] Schema design approved
- [ ] PDF generation library selected (candidates: Puppeteer, PDFKit)
- [ ] Implementation in progress
