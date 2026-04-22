#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  update_url.sh — Change the live domain across the project
#  Usage:  bash update_url.sh
# ─────────────────────────────────────────────────────────────

set -e

OLD_URL="churnprediction.duckdns.org"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Retain.Ai — URL Update Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Current URL: $OLD_URL"
echo ""
read -p "  Enter new domain (e.g. myapp.duckdns.org): " NEW_URL

# Strip protocol if accidentally included
NEW_URL="${NEW_URL#https://}"
NEW_URL="${NEW_URL#http://}"
NEW_URL="${NEW_URL%/}"

if [ -z "$NEW_URL" ]; then
    echo ""
    echo "  ✗  No URL entered. Aborting."
    exit 1
fi

if [ "$NEW_URL" = "$OLD_URL" ]; then
    echo ""
    echo "  ✗  New URL is the same as the old one. Nothing to do."
    exit 0
fi

echo ""
echo "  Replacing: $OLD_URL  →  $NEW_URL"
echo ""

# Detect sed in-place flag (macOS vs Linux differ)
if sed --version 2>/dev/null | grep -q GNU; then
    SED_I="sed -i"
else
    SED_I="sed -i ''"
fi

# ── Files in the repo ─────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

FILES=(
    "$REPO_DIR/README.md"
)

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        $SED_I "s|$OLD_URL|$NEW_URL|g" "$f"
        echo "  ✓  Updated: $(basename $f)"
    fi
done

# ── .env file (local, if it exists) ──────────────────────────
ENV_FILE="$REPO_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    $SED_I "s|$OLD_URL|$NEW_URL|g" "$ENV_FILE"
    echo "  ✓  Updated: .env (local)"
fi

# ── EC2 .env (if running on the server) ──────────────────────
EC2_ENV="/opt/churn_prediction/.env"
if [ -f "$EC2_ENV" ]; then
    if [ -w "$EC2_ENV" ]; then
        $SED_I "s|$OLD_URL|$NEW_URL|g" "$EC2_ENV"
        echo "  ✓  Updated: $EC2_ENV (EC2)"
    else
        echo ""
        echo "  ⚠  EC2 .env found but not writable. Run with sudo:"
        echo "     sudo sed -i 's|$OLD_URL|$NEW_URL|g' $EC2_ENV"
    fi
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done! URL updated to: $NEW_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Next steps:"
echo "  1. Commit and push:  git add -A && git commit -m 'Update domain to $NEW_URL' && git push"
echo "  2. On EC2, update .env manually if not done above:"
echo "       sudo nano /opt/churn_prediction/.env"
echo "       → Update DJANGO_ALLOWED_HOSTS and DJANGO_CSRF_TRUSTED_ORIGINS"
echo "  3. Restart gunicorn:  sudo systemctl restart gunicorn"
echo ""
