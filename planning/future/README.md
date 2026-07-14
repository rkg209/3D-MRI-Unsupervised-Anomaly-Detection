# Archived — NON-NORMATIVE. Do not implement.

The two files here describe a **PostgreSQL database** and a **REST API with authentication**.

**This system has neither.** `planning/02-architecture.md:9` is explicit:

> "This system is a **local research codebase** — not a service, not a deployed application.
> ... **There is no server, no API, no database, and no container orchestration.**"

No spec in `specs/` (000–012) builds a server, a database, migrations, or auth — and none will.

| File | Why it is archived |
|---|---|
| `04-database-schema.sql` | Mandates PostgreSQL 15 + pgcrypto + Flyway migrations + a CI Postgres instance. Also **truncated mid-`SELECT`** — it is not valid SQL and will not execute. |
| `05-openapi.yaml` | OpenAPI 3.0.3 with `servers: localhost:8000/v1` and global `ApiKeyAuth`. Also **truncated mid-response-block** — it is not a parseable OpenAPI document. |

Both appear to be artifacts of a generic planning template that forced a DB doc and an API doc into
a project that needs neither. The prose docs (`04-database-design.md`, `05-api-design.md`) handled
this honestly by framing their SQL/HTTP content as a hypothetical "future projection"; these two
machine-readable files dropped that framing and read as buildable deliverables. They are not.

**The genuinely useful content from those prose docs is kept where it belongs:**
the file-based artifact store (runs, checkpoints, split contract, recon results, metrics) is
specified in `specs/000-vertical-slice.md` and `specs/004-eval-harness.md`, and the exception
hierarchy is a deliverable of Spec 000.

If you are an agent reading this: **do not build a database or an HTTP server.**
