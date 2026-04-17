# AWS EC2 Deployment Guide — Churn Prediction

Zero-cost deployment on AWS Free Tier. Target: **1× EC2 `t2.micro` + local
Postgres + nginx + gunicorn + Let's Encrypt TLS**, all within the 12-month
free tier.

> TL;DR: launch EC2, SSH in, run `sudo bash deploy/scripts/bootstrap_ec2.sh`.

---

## 0. Stop yourself from getting charged

Before you do ANYTHING else in the AWS console:

1. **Billing → Budgets → Create budget** → Zero-spend template → email alert
   at $0.01. Non-negotiable.
2. **Billing preferences** → enable "Receive Free Tier usage alerts".
3. Use a **new AWS account** — free tier is per-account, per-12-months.
4. Pick a region and stay in it. Recommended: `ap-south-1` (Mumbai) for India.

Things that will silently charge you — avoid:

- Elastic IPs on stopped instances
- NAT Gateways ($32/mo)
- Application/Network Load Balancers ($16/mo)
- RDS Multi-AZ, RDS storage > 20 GB
- `t3.micro` in "unlimited" mode burning CPU credits
- Forgetting to delete a snapshot / AMI / EBS volume

---

## 1. Launch the EC2 instance

Console → **EC2 → Launch Instance**:

| Setting | Value |
|---|---|
| Name | `churn-prediction` |
| AMI | Ubuntu Server 24.04 LTS (HVM), SSD Volume Type — **Free tier eligible** |
| Instance type | `t2.micro` (preferred; `t3.micro` also free but watch CPU credits) |
| Key pair | Create new → `churn-key.pem` → download and keep safe |
| Network | Default VPC, auto-assign public IP: **Enabled** |
| Storage | 20 GB gp3 (free tier allows up to 30 GB) |
| Security group | Inbound: SSH (22) from **My IP only**, HTTP (80) from anywhere, HTTPS (443) from anywhere |

Click **Launch**. Note the **Public IPv4 DNS** (looks like
`ec2-13-233-xx-xx.ap-south-1.compute.amazonaws.com`).

---

## 2. Push your code to GitHub

From your laptop:

```bash
cd /path/to/churn_prediction
git init                                  # if not already
git add .
git commit -m "Prep for AWS deploy"
# Create a repo on GitHub, then:
git remote add origin https://github.com/<you>/churn_prediction.git
git push -u origin main
```

Double-check `.gitignore` is in effect — **`db.sqlite3`, `.env`, `myvenv/`,
`__pycache__/` must NOT be in the pushed repo**. Run
`git ls-files | grep -E '(sqlite3|\.env$|myvenv)'` — should print nothing.

If your ML `.pkl` files are in `models/` and tracked in git, that's fine for
now (total ~22 MB). For anything over 100 MB use Git LFS or pull them from
S3 at boot.

---

## 3. SSH into the instance and bootstrap

```bash
chmod 400 churn-key.pem
ssh -i churn-key.pem ubuntu@<YOUR_EC2_PUBLIC_DNS>
```

Then on the server:

```bash
sudo apt-get update && sudo apt-get install -y git
sudo mkdir -p /opt/churn_prediction
sudo chown ubuntu:ubuntu /opt/churn_prediction
git clone https://github.com/<you>/churn_prediction.git /opt/churn_prediction

# Run the bootstrap script. It installs everything and brings the app up.
sudo REPO_URL="" bash /opt/churn_prediction/deploy/scripts/bootstrap_ec2.sh
```

The script:

- Installs Python, nginx, Postgres, certbot, UFW, fail2ban
- Creates a 2 GB swapfile (t2.micro's 1 GB RAM can't build xgboost alone)
- Creates a Python venv and installs `requirements.txt`
- Creates a local Postgres DB and user with a random password
- Generates `.env` with a real `SECRET_KEY` and your EC2 DNS in `ALLOWED_HOSTS`
- Runs `migrate` + `collectstatic`
- Installs and starts the gunicorn systemd service + socket
- Drops the nginx site in place, enables UFW

When it finishes:

```bash
# Create your admin user
cd /opt/churn_prediction
source .venv/bin/activate
set -a; source .env; set +a
python manage.py createsuperuser
deactivate
```

Open `http://<YOUR_EC2_PUBLIC_DNS>/` in your browser. Working? Good.

---

## 4. (Optional, recommended) HTTPS with a free domain

1. Sign up at <https://www.duckdns.org> with Google/GitHub, create a subdomain
   like `yourname-churn.duckdns.org`, and point it at your EC2 public IP.
2. SSH back in:

   ```bash
   sudo certbot --nginx -d yourname-churn.duckdns.org
   ```

   Certbot auto-edits the nginx config to add a 443 server block and an 80→443
   redirect.

3. Update `.env` and restart:

   ```bash
   sudo -i
   nano /opt/churn_prediction/.env
   # DJANGO_ALLOWED_HOSTS=yourname-churn.duckdns.org,<ec2-dns>
   # DJANGO_CSRF_TRUSTED_ORIGINS=https://yourname-churn.duckdns.org
   # DJANGO_SECURE_SSL_REDIRECT=True
   systemctl restart gunicorn
   ```

Auto-renewal is handled by certbot's systemd timer.

---

## 5. Updating the app

Push to GitHub, then on the server:

```bash
sudo bash /opt/churn_prediction/deploy/scripts/deploy.sh
```

---

## 6. If you want to use RDS instead of local Postgres

For the report, RDS looks more "cloud-native." Free tier: `db.t3.micro`
Postgres, 20 GB, Single-AZ, 12 months.

1. RDS → Create database → Postgres → Free tier → Single-AZ → `db.t3.micro`
   → 20 GB gp2 → public access **No** → VPC security group: new SG allowing
   port 5432 **only from the EC2 instance's SG**.
2. Copy endpoint, user, pass, dbname.
3. On EC2, edit `.env`:

   ```
   DATABASE_URL=postgres://<user>:<pass>@<endpoint>:5432/<db>
   DJANGO_DB_SSLMODE=require
   ```
4. `sudo systemctl restart gunicorn && sudo -u www-data /opt/churn_prediction/.venv/bin/python /opt/churn_prediction/manage.py migrate`

**Do not** make the RDS instance publicly accessible. Keep it in the VPC,
reachable only from your EC2 SG.

---

## 7. Architecture for your report

```
                  Internet
                     │
                (HTTPS 443 / HTTP 80)
                     │
          ┌──────────▼───────────┐
          │  EC2 t2.micro        │  ap-south-1a, public subnet
          │  ┌────────────────┐  │
          │  │  nginx         │  │  TLS termination, static, rate-limit
          │  └──────┬─────────┘  │
          │         │ unix sock  │
          │  ┌──────▼─────────┐  │
          │  │ gunicorn       │  │  2 workers × 2 threads, systemd-managed
          │  │   └─ Django    │  │  WhiteNoise, DRF, Jazzmin admin
          │  └──────┬─────────┘  │
          │         │            │
          │  ┌──────▼─────────┐  │
          │  │ Postgres 16    │  │  local, or swap for RDS db.t3.micro
          │  └────────────────┘  │
          │                      │
          │  models/*.pkl on EBS │  scikit-learn, xgboost, joblib
          └──────────────────────┘
                     │
             CloudWatch (logs + billing alarm)
             UFW + fail2ban + SG: 22/80/443 only
```

Services you can legitimately claim in the report:

- **EC2** — compute
- **VPC, Subnets, Security Groups, IAM** — networking + access control
- **EBS** — block storage for the instance
- **Route 53 or DuckDNS** — DNS
- **ACM / Let's Encrypt** — TLS
- **CloudWatch** — monitoring + billing alarms
- **RDS** (if you use it) — managed Postgres
- **S3** (if you move static/models there) — object storage

---

## 8. Security checklist (your cybersec prof will ask)

- [ ] `DEBUG=False` in `.env` (confirmed by `curl -I http://host/does-not-exist`
      returning generic 404, no traceback)
- [ ] `ALLOWED_HOSTS` is specific, not `*`
- [ ] `SECRET_KEY` is random, 50+ chars, only in `.env`, never in git
- [ ] SSH: key-auth only, password auth disabled (default on Ubuntu AMIs)
- [ ] Security Group: SSH restricted to your IP, not `0.0.0.0/0`
- [ ] UFW enabled (bootstrap does this)
- [ ] fail2ban installed (bootstrap does this)
- [ ] HTTPS with HSTS (after certbot step)
- [ ] CSRF + session cookies Secure + HttpOnly (handled by settings.py)
- [ ] RDS (if used) not publicly accessible, TLS required
- [ ] IAM: if accessing S3 from EC2, use an **instance role**, not access keys
- [ ] Admin panel behind a non-default URL or IP allow-list (edit nginx)
- [ ] Run `python manage.py check --deploy` — fix every warning

---

## 9. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `502 Bad Gateway` from nginx | `sudo systemctl status gunicorn`; check `journalctl -u gunicorn -n 200` |
| `DisallowedHost at /` | Add your domain/IP to `DJANGO_ALLOWED_HOSTS` in `.env`, restart gunicorn |
| Static files 404 | `python manage.py collectstatic --noinput`, check `STATIC_ROOT` and nginx `alias` |
| `pip install` killed on t2.micro | Make sure swap is on (`swapon --show`); bootstrap creates 2 GB |
| `.pkl` load error | Python/sklearn version mismatch. Keep pinned versions in `requirements.txt` |
| CSRF error on POST | Add your origin to `DJANGO_CSRF_TRUSTED_ORIGINS` with `https://` scheme |

---

## 10. When the project is over

**Tear down to avoid month-13 charges:**

1. Terminate the EC2 instance
2. Delete the RDS instance (if used) — uncheck "final snapshot"
3. Delete the EBS volume (usually goes with instance)
4. Release any Elastic IPs
5. Empty + delete any S3 buckets
6. Delete the key pair
7. Check **Billing → Cost Explorer** the next day to confirm $0
