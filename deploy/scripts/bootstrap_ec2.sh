#!/usr/bin/env bash
#
# One-shot bootstrap for a fresh Ubuntu 22.04 / 24.04 EC2 instance.
# Run as: sudo bash deploy/scripts/bootstrap_ec2.sh
#
# What it does:
#   - Installs system packages (python, nginx, postgres, certbot, swap).
#   - Creates /opt/churn_prediction, clones-or-copies the app there.
#   - Sets up a Python venv, installs deps, runs migrate + collectstatic.
#   - Installs gunicorn systemd unit + socket, nginx site, enables everything.
#
# After this finishes:
#   - Put your real secrets in /opt/churn_prediction/.env
#   - sudo systemctl restart gunicorn
#   - Hit http://<EC2 PUBLIC IP>/  in a browser.
#
set -euo pipefail

APP_NAME="churn_prediction"
APP_DIR="/opt/${APP_NAME}"
APP_USER="www-data"
PYTHON_BIN="python3"
REPO_URL="${REPO_URL:-}"   # optional: export REPO_URL=https://github.com/you/repo.git

log() { printf '\n\033[1;32m[bootstrap]\033[0m %s\n' "$*"; }

# --- 0. Sanity checks -------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "Run as root (sudo bash $0)" >&2
  exit 1
fi

# --- 1. System packages -----------------------------------------------------
log "Updating apt and installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    build-essential git curl \
    nginx \
    postgresql postgresql-contrib libpq-dev \
    certbot python3-certbot-nginx \
    ufw fail2ban \
    ca-certificates

# --- 2. Swap (1 GB RAM on t2.micro is not enough for pip install of xgboost) -
if ! swapon --show | grep -q '/swapfile'; then
  log "Creating 2 GB swapfile"
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --- 3. App directory -------------------------------------------------------
log "Preparing ${APP_DIR}"
mkdir -p "${APP_DIR}"

if [[ -n "${REPO_URL}" && ! -d "${APP_DIR}/.git" ]]; then
  log "Cloning ${REPO_URL}"
  git clone "${REPO_URL}" "${APP_DIR}"
else
  log "Skipping clone (REPO_URL not set or repo already present)."
  log "If you're copying code manually, scp/rsync into ${APP_DIR} now."
fi

cd "${APP_DIR}"

# --- 4. Python venv + deps --------------------------------------------------
log "Creating virtualenv"
${PYTHON_BIN} -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
deactivate

# --- 5. .env ---------------------------------------------------------------
if [[ ! -f "${APP_DIR}/.env" ]]; then
  log "Creating .env from .env.example (FILL IN REAL VALUES BEFORE STARTING)"
  cp .env.example .env
  # Auto-generate a secret key so the app at least boots.
  SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(50))")
  sed -i "s|^DJANGO_SECRET_KEY=.*|DJANGO_SECRET_KEY=${SECRET}|" .env
  # Put the EC2 public DNS into ALLOWED_HOSTS automatically if available.
  EC2_DNS=$(curl -fs --max-time 2 http://169.254.169.254/latest/meta-data/public-hostname || true)
  if [[ -n "${EC2_DNS}" ]]; then
    sed -i "s|^DJANGO_ALLOWED_HOSTS=.*|DJANGO_ALLOWED_HOSTS=${EC2_DNS}|" .env
  fi
  chmod 640 .env
fi

# --- 6. Postgres (local, free) ---------------------------------------------
# Only runs if DATABASE_URL is empty (i.e. we want local Postgres, not SQLite
# or RDS). Comment this block out if you prefer SQLite for the demo.
if grep -q '^DATABASE_URL=$' .env; then
  log "Setting up local Postgres (comment out this block to keep SQLite)"
  DB_PASS=$(python3 -c "import secrets;print(secrets.token_urlsafe(24))")
  sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='churn_user'" | grep -q 1 || \
      sudo -u postgres psql -c "CREATE ROLE churn_user LOGIN PASSWORD '${DB_PASS}';"
  sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='churn_db'" | grep -q 1 || \
      sudo -u postgres createdb -O churn_user churn_db
  sed -i "s|^DATABASE_URL=$|DATABASE_URL=postgres://churn_user:${DB_PASS}@127.0.0.1:5432/churn_db|" .env
fi

# --- 7. Django setup (migrate + collectstatic) ------------------------------
log "Running migrate + collectstatic"
source .venv/bin/activate
set -a; source .env; set +a
python manage.py migrate --noinput
python manage.py collectstatic --noinput
deactivate

# --- 8. Permissions ---------------------------------------------------------
log "Fixing permissions"
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
chmod 750 "${APP_DIR}"
chmod 640 "${APP_DIR}/.env"

# --- 9. systemd -------------------------------------------------------------
log "Installing systemd units"
install -m 0644 "${APP_DIR}/deploy/systemd/gunicorn.socket"  /etc/systemd/system/gunicorn.socket
install -m 0644 "${APP_DIR}/deploy/systemd/gunicorn.service" /etc/systemd/system/gunicorn.service
systemctl daemon-reload
systemctl enable --now gunicorn.socket
systemctl enable --now gunicorn.service

# --- 10. nginx --------------------------------------------------------------
log "Installing nginx site"
# Inject EC2 DNS into server_name if we can.
EC2_DNS=$(curl -fs --max-time 2 http://169.254.169.254/latest/meta-data/public-hostname || echo "_")
sed "s|SERVER_NAME_PLACEHOLDER|${EC2_DNS}|" \
    "${APP_DIR}/deploy/nginx/churn_prediction.conf" \
    > /etc/nginx/sites-available/churn_prediction
ln -sf /etc/nginx/sites-available/churn_prediction /etc/nginx/sites-enabled/churn_prediction
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

# --- 11. Firewall -----------------------------------------------------------
log "Configuring UFW"
ufw allow OpenSSH || true
ufw allow 'Nginx Full' || true
yes | ufw enable || true

# --- 12. Done ---------------------------------------------------------------
log "Bootstrap complete."
echo
echo "Next:"
echo "  1. Edit /opt/churn_prediction/.env  (ALLOWED_HOSTS, etc.)"
echo "  2. sudo python3 /opt/churn_prediction/manage.py createsuperuser"
echo "     (run it inside the venv: source /opt/churn_prediction/.venv/bin/activate)"
echo "  3. Visit: http://${EC2_DNS}/"
echo "  4. For HTTPS: sudo certbot --nginx -d <your-domain>"
