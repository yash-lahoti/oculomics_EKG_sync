#!/bin/zsh
# setup-mac.sh — automate Steps 1–5 of LOCAL_SETUP.md:
# Homebrew PHP 8.3 + MariaDB, database creation, Leantime release download, config/.env.
#
# NOTE: written and syntax-checked on Linux, not executed on a real Mac.
# If any step fails, stop and follow the manual steps in LOCAL_SETUP.md — they are canonical.
#
# Usage:            ./setup-mac.sh
# Custom version:   LT_VERSION=3.9.9 ./setup-mac.sh
# Custom location:  INSTALL_DIR=$HOME/apps/leantime ./setup-mac.sh

set -euo pipefail

LT_VERSION="${LT_VERSION:-3.9.8}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/Leantime}"
PORT="${PORT:-8080}"
DB_NAME="${DB_NAME:-leantime}"
DB_USER="${DB_USER:-leantime}"
DB_PASS="${DB_PASS:-$(openssl rand -hex 16)}"

step()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail()  { printf '\033[31mERROR: %s\033[0m\nFall back to the manual steps in LOCAL_SETUP.md.\n' "$1" >&2; exit 1; }

command -v brew >/dev/null 2>&1 || fail "Homebrew not found — install it from https://brew.sh first"

step "Installing PHP 8.3 and MariaDB (skips anything already installed)"
brew list php@8.3 >/dev/null 2>&1 || brew install php@8.3
brew list mariadb >/dev/null 2>&1 || brew install mariadb
PHP_BIN="$(brew --prefix php@8.3)/bin/php"
[ -x "$PHP_BIN" ] || fail "PHP not found at $PHP_BIN"

step "Starting MariaDB (registers it to start at login)"
brew services start mariadb >/dev/null

MARIADB_BIN="$(command -v mariadb || command -v mysql || true)"
[ -n "$MARIADB_BIN" ] || fail "mariadb/mysql client not on PATH — open a new terminal and re-run"

printf 'Waiting for MariaDB to accept connections'
for i in {1..30}; do
  if "$MARIADB_BIN" -u root -e 'SELECT 1' >/dev/null 2>&1; then break; fi
  [ "$i" -eq 30 ] && fail "MariaDB did not come up within 30s (try: brew services restart mariadb)"
  printf '.'; sleep 1
done
printf ' up.\n'

step "Creating database '$DB_NAME' and user '$DB_USER'"
"$MARIADB_BIN" -u root <<SQL
CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '$DB_USER'@'localhost' IDENTIFIED BY '$DB_PASS';
ALTER USER '$DB_USER'@'localhost' IDENTIFIED BY '$DB_PASS';
GRANT ALL PRIVILEGES ON \`$DB_NAME\`.* TO '$DB_USER'@'localhost';
FLUSH PRIVILEGES;
SQL

step "Downloading Leantime v$LT_VERSION release package"
if [ -f "$INSTALL_DIR/config/.env" ]; then
  fail "$INSTALL_DIR already contains a configured Leantime — refusing to overwrite. For updates, see Step 7 of LOCAL_SETUP.md"
fi
mkdir -p "$INSTALL_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ZIP_URL="https://github.com/Leantime/leantime/releases/download/v$LT_VERSION/Leantime-v$LT_VERSION.zip"
curl -fL --retry 3 -o "$TMP_DIR/leantime.zip" "$ZIP_URL" \
  || fail "download failed — grab Leantime-v$LT_VERSION.zip manually from https://github.com/Leantime/leantime/releases"

step "Unpacking to $INSTALL_DIR"
unzip -q "$TMP_DIR/leantime.zip" -d "$TMP_DIR/unpacked"
# The archive may contain the files directly or a single wrapper directory — handle both.
if [ -d "$TMP_DIR/unpacked/public" ]; then
  SRC="$TMP_DIR/unpacked"
else
  SRC="$(find "$TMP_DIR/unpacked" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [ -n "$SRC" ] && [ -d "$SRC/public" ] || fail "unexpected archive layout — unzip manually per Step 3 of LOCAL_SETUP.md"
fi
cp -R "$SRC/." "$INSTALL_DIR/"
[ -f "$INSTALL_DIR/bin/leantime" ] || fail "copy failed — $INSTALL_DIR/bin/leantime missing"

step "Writing config/.env"
cp "$INSTALL_DIR/config/sample.env" "$INSTALL_DIR/config/.env"
SESSION_PW="$(openssl rand -hex 32)"
patch_env() {  # patch_env KEY VALUE — replaces the key's line in config/.env (BSD sed)
  sed -i '' "s|^[[:space:]]*$1[[:space:]]*=.*|$1 = \"$2\"|" "$INSTALL_DIR/config/.env"
}
patch_env LEAN_APP_URL          "http://localhost:$PORT"
patch_env LEAN_SITENAME         "Residency Planner"
patch_env LEAN_SESSION_PASSWORD "$SESSION_PW"
patch_env LEAN_DB_HOST          "localhost"
patch_env LEAN_DB_PORT          "3306"
patch_env LEAN_DB_USER          "$DB_USER"
patch_env LEAN_DB_PASSWORD      "$DB_PASS"
patch_env LEAN_DB_DATABASE      "$DB_NAME"

step "Done. Next steps"
cat <<NEXT

  1. Start Leantime:
       cd $INSTALL_DIR && "$PHP_BIN" bin/leantime serve --port=$PORT

  2. Open http://localhost:$PORT/install and create your admin account.

  3. Auto-start at login + reminders + backups: Steps 6–7 of LOCAL_SETUP.md.

  Database password for '$DB_USER' (also saved in $INSTALL_DIR/config/.env):
       $DB_PASS

NEXT
