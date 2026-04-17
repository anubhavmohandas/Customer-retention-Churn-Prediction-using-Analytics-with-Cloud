"""
Gunicorn config for churn_prediction.
Tuned for a free-tier t2.micro / t3.micro (1 vCPU, 1 GB RAM).
"""
import multiprocessing
import os

bind = os.environ.get('GUNICORN_BIND', 'unix:/run/gunicorn/churn.sock')

# On a 1 vCPU box, (2 * cores) + 1 = 3 workers is too many with ML libs loaded.
# Keep it small to stay under ~700 MB RAM.
workers = int(os.environ.get('GUNICORN_WORKERS', '2'))
threads = int(os.environ.get('GUNICORN_THREADS', '2'))
worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'gthread')

timeout = int(os.environ.get('GUNICORN_TIMEOUT', '60'))
graceful_timeout = 30
keepalive = 5

# Recycle workers to combat memory leaks from pandas/xgboost.
max_requests = 500
max_requests_jitter = 50

# Logs go to stdout/stderr -> systemd journal -> CloudWatch if agent installed.
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Preload to share read-only memory (model pickles) across workers.
preload_app = True
umask = 0o007
