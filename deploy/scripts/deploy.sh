#!/usr/bin/env bash
#
# Pull latest code, update deps, migrate, collectstatic, restart gunicorn.
# Run as: sudo bash /opt/churn_prediction/deploy/scripts/deploy.sh
#
set -euo pipefail

APP_DIR="/opt/churn_prediction"
cd "${APP_DIR}"

echo "[deploy] git pull"
sudo -u www-data git pull --ff-only || git pull --ff-only

echo "[deploy] pip install"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -r requirements.txt
set -a; source .env; set +a

echo "[deploy] migrate"
python manage.py migrate --noinput

echo "[deploy] collectstatic"
python manage.py collectstatic --noinput

deactivate

echo "[deploy] restart gunicorn"
systemctl restart gunicorn

echo "[deploy] done."
systemctl --no-pager --lines=10 status gunicorn
