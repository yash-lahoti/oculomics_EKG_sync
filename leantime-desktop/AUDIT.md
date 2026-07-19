# Leantime v3.9.8 — Database-Portability & Desktop-Feasibility Audit

**Audited:** `Leantime/leantime` tag `v3.9.8` (commit `32be54e`, released 2026-07-08 — latest stable at audit time, 2026-07-19).
**Method:** static analysis of a fresh clone. Every count below has its command in the [Appendix](#appendix-reproduce-every-number) so you can re-run it.
**File references** are paths inside the Leantime repo.

---

## Headline: the analysis this audit was commissioned against is out of date — in your favor

The migration plan you started from assumed Leantime was a MySQL-only codebase full of raw SQL, and priced the SQLite conversion at **40–60% of a 4–10 week effort**. That was true of older Leantime. It is no longer true of v3.9.8:

1. **The database-abstraction refactor already happened upstream.** The app now runs on Laravel Query Builder (349 `->table(` call sites) with only **6 raw `->prepare(` statements left in the entire application**. A dedicated compatibility layer (`app/Core/Db/DatabaseHelper.php`) translates every remaining dialect-specific function per driver, and Leantime officially targets **MySQL, PostgreSQL, and MS SQL Server**. A `sqlite` connection block already exists in the config (`app/Core/Configuration/laravelConfig.php:558`) selectable via `LEAN_DB_DEFAULT_CONNECTION`.

2. **Fresh installs no longer run raw MySQL DDL.** The installer calls `SchemaBuilder->createAllTables()` — Laravel's database-agnostic schema builder (`app/Domain/Install/Repositories/Install.php:261-264`). The infamous 2,845-line `Install.php` with 41 raw `CREATE TABLE`s and 44 `update_sql_*` methods is now the *upgrade* path for existing databases, not the install path.

3. **"Phase 5: Add MCP" already shipped.** Leantime 3.9.7+ bundles the official `laravel/mcp` package (composer.json: `laravel/mcp ^0.1.1`), exposes an authenticated **`/mcp` endpoint** (`app/Core/Http/IncomingRequest.php:203-209`, own rate-limit budget at `app/Core/Middleware/RequestRateLimiter.php:97`), and ships **55 MCP tools** across 6 domains — including `AddTaskTool`, `EditTaskTool`, `BulkAddTasksTool`, `FindMilestonesTool`, `GetCalendarTool`, `GetFullProjectOverviewTool`, `AddCommentTool`, `CreateGoalTool`, `ScheduleDayTool`. That is a superset of the `create_task` / `update_task` / `list_upcoming_deadlines` / `summarize_application_status` tool list you sketched. Tokens are minted with `php bin/leantime auth:create-bearer-token --email=<user>`.

**Consequence:** for your actual goal (a personal residency-application organizer with agent integration), **no migration is required at all** — see `LOCAL_SETUP.md` in this folder. The SQLite question only matters if you later want a distributable desktop product, and even then it has shrunk from "rewrite the data layer" to "extend an existing pattern."

---

## 1. Raw-SQL inventory

| Metric | Count | Where |
|---|---|---|
| Repository classes | 34 files | `app/Domain/*/Repositories/`, `app/Core/Db/Repository.php` |
| Laravel Query Builder call sites (`->table(`) | **349** | throughout domains |
| Raw PDO `->prepare(` statements | **6** | `app/Core/Db/Repository.php` ×4 (generic base-repo helpers), `app/Domain/Timesheets/Repositories/Timesheets.php` ×1, `app/Domain/Install/Repositories/Install.php` ×1 |
| Eloquent models | 0 | plain Query Builder + typed repositories by design |
| `DB::` facade uses | 15 | mostly install/maintenance code |
| Direct `mysqli_*` calls | **0** | `ext-mysqli` is still a composer requirement (legacy constraint), but nothing in `app/` calls it |

## 2. MySQL-specific constructs remaining

| Construct | Occurrences in `app/` | Assessment |
|---|---|---|
| `GROUP_CONCAT`, `DATE_FORMAT`, `WEEK()`, `CURDATE`, `NOW()`, `FIND_IN_SET`, `IFNULL`, `INTERVAL` | 42 total, concentrated in 5 files | **Nearly all inside per-driver `match` blocks.** `DatabaseHelper.php` (10) holds the central translations; `Timesheets.php` (23), `Tickets.php` (4), `Projects.php` (2), `Install.php` (3) contain inline `match ($driver)` blocks of the same shape — e.g. `Timesheets.php:345-349` emits `DATE_FORMAT` for mysql, `TO_CHAR` for pgsql |
| `ON DUPLICATE KEY UPDATE` / `INSERT IGNORE` | 7 matches | 4 are comments documenting that the MySQL-only upsert was **already replaced** with read-then-write or per-driver dialect (`Timesheets.php:604,711,734,776`); 2 live only in the legacy install/upgrade path (`Install.php:1459,2264`); 1 commented-out (`Install.php:356`) |
| `ENUM(` columns | **0** | — |
| `FULLTEXT` / `MATCH (` | **0** | — |
| `JSON_*` functions | **0** | — |
| Legacy MySQL DDL | 41 `CREATE TABLE` + 44 `update_sql_*` methods | **Upgrade path only** — fresh installs use `SchemaBuilder` (988 lines, Laravel `Schema::` API, already handles pgsql sequence resync at `SchemaBuilder.php:137`) |

### The DatabaseHelper pattern (what a SQLite port would extend)

`app/Core/Db/DatabaseHelper.php` centralizes 12 dialect helpers (`stringAggregate`, `weekNumber`, `formatDate`, `yesterdayDate`, `currentDate`, `isYesterday`, `currentTimestamp`, `findInSet`, `ifNull`, `ifThen`, `wrapColumn`, `castAs`). Each is a `match` on `getDriverName()` with `mysql` / `pgsql` / `sqlsrv` arms and a MySQL fallback. **There is no `sqlite` arm anywhere** — that is the concrete gap. Every helper has a known SQLite equivalent (`group_concat(x, ',')`, `strftime`, `date('now','-1 day')`, `instr`-based find-in-set, `COALESCE`, etc.), so each arm is a few lines.

## 3. Multi-driver support: how deep it actually goes

- Connection config defines `sqlite`, `mysql`, `pgsql` (+ custom `LtPostgresConnection` class in `app/Core/Database/`); default driver comes from `LEAN_DB_DEFAULT_CONNECTION` (`laravelConfig.php:546-567`).
- The installer branches per driver — `mysql` and `pgsql` are handled explicitly (`Install.php:151-165`, `:230-238`); **no `sqlite` branch exists**, so today the `/install` wizard cannot complete on SQLite.
- Sessions, cache, and the queue all ride the same Laravel database connection (queue default driver `database` on table `zp_jobs`, `laravelConfig.php:915,936-943`) — nothing separately MySQL-bound.

## 4. Everything else the desktop plan worried about

| Concern from your plan | Reality in v3.9.8 |
|---|---|
| Installer workflow | Web wizard at `/install`; upgrades via `/update` route **or** `php bin/leantime system:update`; full artisan-style CLI (`bin/leantime`) incl. `db:backup`, `user add`, `migrate` |
| Cron / background jobs | Laravel scheduler. Trigger = `curl /cron/run` **or** `php bin/leantime schedule:run` (`app/Domain/Cron/Services/Cron.php:44`); queue is DB-backed, no daemon required |
| Filesystem assumptions | Laravel filesystems: local disk rooted at `<app>/userfiles` + `public/userfiles`; S3 optional (`LEAN_USE_S3`, off by default); logs/backups/userfiles relocatable via `LEAN_LOG_PATH`, `LEAN_DB_BACKUP_PATH`, `LEAN_USER_FILE_PATH` — i.e., an app-data directory layout is a config change, not a code change |
| Authentication excess | LDAP / OIDC / SAML are **config-gated and off by default**; a plain local admin account is the default path. Nothing to rip out for personal use |
| PHP runtime | `php ^8.2` (CI runs 8.3); 15 required extensions incl. `pdo_mysql`, `mysqli`, `ldap`, `pcntl`, `posix`, `gd` — all present in a standard Homebrew PHP build. For NativePHP packaging, `ldap`/`pcntl`/`posix` availability in the bundled static PHP is a real item to verify |
| Frontend / assets | Release package (`Leantime-vX.Y.Z.zip`) ships `vendor/` + prebuilt assets; docroot is `public/`; `php bin/leantime serve` is a registered command (`app/Core/Console/CliServiceProvider.php:206`) |

## 5. Classification (your rubric)

| Classification | What falls in it | Volume |
|---|---|---|
| **Portable** — works unchanged on SQLite | 349 Query-Builder call sites; SchemaBuilder fresh-install DDL (Laravel `Schema::` API); sessions/cache/queue (Laravel DB drivers support SQLite); auth; file storage; plugins framework | ~95% of the data layer |
| **Small modification** | Add `sqlite` arms: 12 `DatabaseHelper` methods + ~29 inline `match($driver)` sites in Timesheets/Tickets/Projects/Install; add a `sqlite` branch to the installer (skip CREATE DATABASE/USE, set FK pragma); default-connection plumbing already exists | ~40–45 small, mechanical edits |
| **Replacement required** | The 44 `update_sql_*` MySQL upgrade scripts — **only if you need to migrate an existing MySQL database's history to SQLite**. A fresh SQLite install bypasses them entirely via SchemaBuilder (that's already how the PostgreSQL path works) | 0 for fresh installs |
| **Optional feature** | LDAP, OIDC, S3, Redis, SMTP email, rate-limiting knobs — all env-gated, all default-off | disable-by-ignoring |

**Revised estimate for a SQLite-capable browser version:** ~**1–2 weeks** for an experienced developer including regression testing of the major flows — not the 4–10 weeks in the original plan. The pgsql/sqlsrv work upstream also signals that portability is an active goal there; before doing this yourself, check the Leantime issue tracker — SQLite may land upstream, and the helper-arm pattern makes a clean PR if you want to contribute it.

The parts of the original analysis that **still stand**: desktop *packaging* (NativePHP window, first-run init, app-data dirs, signing, notarization, updates) remains a 1–2 month polish effort for a distributable product, and Path B (SQLite) remains the right architecture for one. Path A (bundling MariaDB) is now strictly dominated — the SQLite gap is too small to justify shipping a database server.

## 6. Licensing

Leantime is AGPL-3.0. **Running it locally for yourself is not "conveying" under AGPL — no obligations are triggered.** The source-availability requirements only bind you if you distribute the app (or run it as a network service for others). For the personal-use path in `LOCAL_SETUP.md`, licensing is a non-issue. A closed-source commercial product would still need the clean-room approach from your original analysis.

## 7. Recommendation

1. **Now:** run stock Leantime natively on your Mac (`LOCAL_SETUP.md`, ~20 minutes). You get the full feature set, local-only data, and a working `/mcp` endpoint for Claude — zero code changes, zero AGPL exposure.
2. **If the desktop itch returns:** the SQLite conversion is a 1–2 week, well-bounded change (Section 5). Watch/ask upstream first.
3. **Skip entirely:** Path A (bundled MariaDB desktop build), pre-emptive auth surgery, and any Tauri rewrite — the reimplementation-vs-migrate tradeoff flipped once the data layer and MCP turned out to be done.

---

## Appendix: reproduce every number

Run from a Leantime checkout at `v3.9.8` (`git clone https://github.com/Leantime/leantime && cd leantime && git checkout v3.9.8`). Counts use ripgrep-compatible patterns; plain `grep -rn` works too.

```bash
# Repository classes
find app -path '*Repositories*' -name '*.php' | wc -l                          # 34

# Query Builder vs raw PDO
grep -rn -- '->table(' app | wc -l                                             # 349
grep -rn -- '->prepare(' app | grep -c '\.php'                                 # 6

# Eloquent / facade / mysqli
grep -rln 'extends Model' app | wc -l                                          # 0
grep -rn 'DB::' app | wc -l                                                    # 15
grep -rn 'mysqli_' app | wc -l                                                 # 0

# MySQL-specific functions (concentration)
grep -rlc 'GROUP_CONCAT\|ON DUPLICATE KEY\|INSERT IGNORE\|DATE_FORMAT\|STR_TO_DATE\|FROM_UNIXTIME\|UNIX_TIMESTAMP' app
# -> DatabaseHelper.php:10, Timesheets.php:23+, Tickets.php:4, Projects.php:2, Install.php:3

# Constructs that are simply absent
grep -rno 'ENUM(\|FULLTEXT\|JSON_[A-Z]*(\|MATCH (' app | wc -l                 # 0

# Multi-driver evidence
grep -n "sqlite" app/Core/Configuration/laravelConfig.php                      # sqlite connection block
grep -rn 'sqlite\|pgsql\|sqlsrv' app/Core/Db/DatabaseHelper.php | wc -l        # 24 (no sqlite arm among them)
grep -n 'SchemaBuilder' app/Domain/Install/Repositories/Install.php            # fresh installs use SchemaBuilder

# Legacy upgrade path size
grep -c 'CREATE TABLE' app/Domain/Install/Repositories/Install.php             # 41
grep -c 'function update_sql' app/Domain/Install/Repositories/Install.php      # 44

# MCP surface
find app -path '*Tools*' -name '*Tool.php' | wc -l                             # 55
grep -n "'/mcp'" app/Core/Http/IncomingRequest.php
python3 -c "import json; print(json.load(open('composer.json'))['require']['laravel/mcp'])"   # ^0.1.1

# Cron & CLI
grep -n "schedule:run" app/Domain/Cron/Services/Cron.php
grep -rn "name: 'db:backup'\|name: 'system:update'\|name: 'auth:create-bearer-token'" app/Command/
```
