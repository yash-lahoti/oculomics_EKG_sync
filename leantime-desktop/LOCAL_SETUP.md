# Run Leantime Locally on Your Mac (No Docker)

This gets the exact same Leantime that runs on servers — every feature, including the built-in MCP endpoint for Claude — running natively on macOS, storing everything on your own disk. Rough time: **20–30 minutes**. No Docker, no cloud, no account with anyone.

Verified against **Leantime v3.9.8** (July 2026). The commands assume the default zsh shell.

> **Shortcut:** `setup-mac.sh` in this folder automates Steps 1–5. It follows exactly the steps below, but it was written on Linux and not executed on a real Mac — if anything errors, fall back to the manual steps, which are the source of truth.

---

## Step 1 — Install PHP and MariaDB with Homebrew

If you don't have Homebrew, install it first from [brew.sh](https://brew.sh).

```bash
brew install php@8.3 mariadb
brew services start mariadb
```

- **Why `php@8.3`:** Leantime requires PHP ≥ 8.2 and its CI tests on 8.3. Homebrew's plain `php` may be newer than what Leantime tests against — the pinned formula avoids surprises.
- `php@8.3` is "keg-only," meaning it isn't on your PATH by default. That's fine — we'll call it by full path. Set a convenience variable (add this line to `~/.zshrc` to make it permanent):

```bash
export LEAN_PHP="$(brew --prefix php@8.3)/bin/php"
$LEAN_PHP -v   # should print PHP 8.3.x
```

- `brew services start mariadb` also registers MariaDB to **start automatically at login**, so your data is always available.

## Step 2 — Create the database

Homebrew's MariaDB lets your macOS user in as root without a password:

```bash
mariadb -u root <<'SQL'
CREATE DATABASE IF NOT EXISTS leantime CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'leantime'@'localhost' IDENTIFIED BY 'CHOOSE_A_DB_PASSWORD';
GRANT ALL PRIVILEGES ON leantime.* TO 'leantime'@'localhost';
FLUSH PRIVILEGES;
SQL
```

Replace `CHOOSE_A_DB_PASSWORD` with something you'll paste into the config in Step 4. (If `mariadb` isn't found, the client may be named `mysql` on your install — same commands.)

## Step 3 — Download the Leantime release package

Get the latest release from **https://github.com/Leantime/leantime/releases** — the file is named `Leantime-vX.Y.Z.zip` (v3.9.8 at time of writing). The package ships with all dependencies and prebuilt assets, so there is **no composer or npm step**.

```bash
mkdir -p ~/Leantime && cd ~/Leantime
curl -fL -o leantime.zip \
  https://github.com/Leantime/leantime/releases/download/v3.9.8/Leantime-v3.9.8.zip
unzip -q leantime.zip && rm leantime.zip
# If the zip extracted into a nested folder (ls shows one directory instead of app/, bin/, public/...):
#   mv <that-folder>/* <that-folder>/.* . 2>/dev/null; rmdir <that-folder>
ls   # expect: app/  bin/  bootstrap/  config/  public/  vendor/  ...
```

## Step 4 — Configure

```bash
cd ~/Leantime
cp config/sample.env config/.env
```

Open `config/.env` in any editor and set the values below (or copy `.env.sample` from this folder over it and fill in the two passwords — it contains only the keys that matter for a local single-user install):

```ini
LEAN_APP_URL = "http://localhost:8080"
LEAN_SITENAME = "Residency Planner"
LEAN_DEFAULT_TIMEZONE = "America/New_York"

LEAN_SESSION_PASSWORD = "<paste output of: openssl rand -hex 32>"

LEAN_DB_HOST = "localhost"
LEAN_DB_USER = "leantime"
LEAN_DB_PASSWORD = "CHOOSE_A_DB_PASSWORD"   # from Step 2
LEAN_DB_DATABASE = "leantime"
LEAN_DB_PORT = "3306"
```

Everything else in the file (S3, LDAP, OIDC, Redis, SMTP…) is optional and off by default — leave it alone.

## Step 5 — Start it and run the install wizard

```bash
cd ~/Leantime
$LEAN_PHP bin/leantime serve --port=8080
```

Open **http://localhost:8080/install** — the wizard creates the database tables and your admin account. Then log in at http://localhost:8080. That's the entire technical setup; the terminal window just needs to stay open (Step 6 fixes that).

## Step 6 — Make it feel like an app (auto-start + Dock icon)

**Auto-start at login.** Copy `com.leantime.server.plist` from this folder:

```bash
cp com.leantime.server.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.leantime.server.plist
```

Leantime is now always running at http://localhost:8080 whenever you're logged in (server log: `~/Library/Logs/leantime-server.log`). To stop it: `launchctl unload ~/Library/LaunchAgents/com.leantime.server.plist`.

**Dock icon.** In Safari, open http://localhost:8080 → **File → Add to Dock** (or in Chrome: ⋮ → Cast, Save and Share → Install page as app). You get a standalone window with its own icon — a desktop app in every way that matters day-to-day, with none of the packaging work.

**Reminders & scheduled jobs.** Leantime's reminders/digests run off a scheduler. Give it a heartbeat (runs only while your Mac is awake, which is fine for a personal tool):

```bash
crontab -e
# add this line (single line, adjust username):
* * * * * cd $HOME/Leantime && $(brew --prefix php@8.3)/bin/php bin/leantime schedule:run >> $HOME/Library/Logs/leantime-cron.log 2>&1
```

In-app notifications work regardless. Email notifications additionally need the SMTP settings in `config/.env` — optional.

## Step 7 — Backups, updates, uninstall

Everything lives in exactly three places: the **database**, the **`userfiles/` folders** (attachments), and **`config/.env`**.

**Backup** (put it in a weekly cron line, or run before updates):

```bash
mkdir -p ~/LeantimeBackups
mariadb-dump -u leantime -p leantime > ~/LeantimeBackups/leantime-$(date +%F).sql
cp -R ~/Leantime/userfiles ~/Leantime/public/userfiles ~/Leantime/config/.env ~/LeantimeBackups/
```

(Leantime also has a built-in `$LEAN_PHP bin/leantime db:backup`, and Time Machine covers all of `~/Leantime` automatically.)

**Restore:** recreate the database (Step 2), `mariadb -u leantime -p leantime < backup.sql`, copy the folders back.

**Update** when a new release comes out:

```bash
# 1. back up (above)   2. download the new release zip   3. then:
cd ~/Leantime
unzip -oq ~/Downloads/Leantime-vX.Y.Z.zip    # overwrites code; userfiles/ and config/.env are untouched
$LEAN_PHP bin/leantime system:update          # runs any database migrations (or visit /update)
```

**Uninstall completely:**

```bash
launchctl unload ~/Library/LaunchAgents/com.leantime.server.plist
rm ~/Library/LaunchAgents/com.leantime.server.plist
crontab -e            # remove the schedule:run line
mariadb -u root -e "DROP DATABASE leantime; DROP USER 'leantime'@'localhost';"
rm -rf ~/Leantime
brew services stop mariadb   # plus `brew uninstall mariadb php@8.3` if nothing else uses them
```

---

## Organizing your residency application in it

A structure that maps cleanly onto Leantime's features (adapt freely):

**One project: "Residency Applications 2026–27".**

- **Milestones** (they render on the timeline/Gantt and drive deadline views) — enter the real dates from the current [ERAS calendar](https://students-residents.aamc.org/applying-residencies-eras/eras-tools-and-worksheets-residency-applicants) and [NRMP calendar](https://www.nrmp.org); the season's anchors are typically: ERAS opens for applicant registration → application submission opens (late Sep) → MSPE released (Oct 1) → interview season (Nov–Jan) → rank-order list opens and certification deadline (late Feb/early Mar 2027) → SOAP → **Match Day (mid-Mar 2027)**.
- **One task per program**, on a board with columns like `Researching → Applying → Applied → Interview offered → Interviewed → Ranked`. Use **subtasks** on each program for the repeatable checklist: supplemental/secondary, letters assigned, signal used (y/n), thank-you sent.
- **To-dos with due dates** for one-off work: personal statement drafts, transcript request, photo, LoR reminders to writers.
- **Goals** for the numbers you're tracking: "X applications submitted by Sep 25," "Y interviews scheduled."
- **Calendar** for interview dates (it also exposes an iCal URL — subscribe from Apple Calendar so interviews appear on your phone).
- **Docs/Wiki** for program research notes, and an **Idea board** for the maybe-list of programs.

## Optional: connect Claude to it (built-in MCP)

Your local Leantime exposes an MCP server at `http://localhost:8080/mcp` with 55 tools (create/edit tasks, milestones, calendar, goals, comments, project overviews). To use it from an MCP client:

```bash
cd ~/Leantime
$LEAN_PHP bin/leantime auth:create-bearer-token --email=you@yourlogin --name=claude
```

Then register it in your MCP client. For Claude Desktop, add to `claude_desktop_config.json` (uses the `mcp-remote` bridge for header auth; requires Node):

```json
{
  "mcpServers": {
    "leantime": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8080/mcp",
               "--header", "Authorization: Bearer PASTE_TOKEN_HERE"]
    }
  }
}
```

After restarting the client, "add my interview at Program X on Dec 4 and a thank-you-note task for the day after" becomes a one-line request. The token grants full access as your user — treat it like a password (it never leaves your machine; the server only listens on localhost).

---

## If something goes wrong

- **Blank page / 500:** set `LEAN_DEBUG = "1"` in `config/.env`, reload, read the error; logs are in `~/Leantime/storage/logs/`.
- **"Connection refused" to DB:** `brew services list` — MariaDB must show `started`. Then re-check the four `LEAN_DB_*` values.
- **Port 8080 taken:** pick another port in both the serve command/plist and `LEAN_APP_URL`.
- **After an update the app won't boot:** delete `~/Leantime/bootstrap/cache/*.php` and reload (stale compiled caches; v3.9.8 handles this automatically for the MCP provider case).
