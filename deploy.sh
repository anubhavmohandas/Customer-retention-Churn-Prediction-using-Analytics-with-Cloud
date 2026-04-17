#sudo tee /opt/deploy.sh > /dev/null << 'EOF'
#!/bin/bash
set -euo pipefail

APP_DIR="/opt/churn_prediction"
VENV="$APP_DIR/.venv/bin"
ENV_FILE="$APP_DIR/.env"

echo "========================================"
echo "  RetainIQ Deploy — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Save HEAD before pull
PREV_HEAD=$(sudo -u www-data git -C "$APP_DIR" rev-parse HEAD)

# Pull
echo ""
echo ">>> Pulling from GitHub..."
sudo -u www-data git -C "$APP_DIR" pull

NEW_HEAD=$(sudo -u www-data git -C "$APP_DIR" rev-parse HEAD)

if [ "$PREV_HEAD" = "$NEW_HEAD" ]; then
    echo "Already up to date — nothing to deploy."
    exit 0
fi

# What changed?
CHANGED=$(sudo -u www-data git -C "$APP_DIR" diff --name-only "$PREV_HEAD" "$NEW_HEAD")
echo ""
echo ">>> Changed files:"
echo "$CHANGED"

# requirements.txt
if echo "$CHANGED" | grep -q "requirements.txt"; then
    echo ""
    echo ">>> requirements.txt changed — installing packages..."
    sudo -u www-data "$VENV/pip" install -r "$APP_DIR/requirements.txt"
fi

# models / migrations
if echo "$CHANGED" | grep -qE "models\.py|migrations/"; then
    echo ""
    echo ">>> Models changed — running migrations..."
    sudo -u www-data -H bash -c \
        "set -a && . $ENV_FILE && set +a && cd $APP_DIR && $VENV/python manage.py migrate"
fi

# static assets (CSS/JS/images — not templates)
if echo "$CHANGED" | grep -qE "^static/"; then
    echo ""
    echo ">>> Static files changed — collecting static..."
    sudo -u www-data -H bash -c \
        "set -a && . $ENV_FILE && set +a && cd $APP_DIR && $VENV/python manage.py collectstatic --noinput"
fi

# settings.py — run deploy check
if echo "$CHANGED" | grep -q "settings.py"; then
    echo ""
    echo ">>> settings.py changed — running deploy check..."
    sudo -u www-data -H bash -c \
        "set -a && . $ENV_FILE && set +a && cd $APP_DIR && $VENV/python manage.py check --deploy" \
        || echo "WARNING: deploy check returned issues — review before proceeding"
fi

# Always restart gunicorn
echo ""
echo ">>> Restarting gunicorn..."
sudo systemctl restart gunicorn

echo ""
echo ">>> Status:"
sudo systemctl status gunicorn --no-pager | head -6

echo ""
echo "========================================"
echo "  Deploy complete — $(git -C $APP_DIR log -1 --format='%h %s')"
echo "========================================"
#EOF

sudo chmod +x /opt/deploy.sh
echo "Done — script at /opt/deploy.sh"