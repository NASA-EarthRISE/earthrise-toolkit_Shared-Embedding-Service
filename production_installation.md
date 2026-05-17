# Production Installation Guide

## Overview

The Shared Embedding Service runs on **Gunicorn with Uvicorn workers**, fronted
by **Nginx**, managed by **systemd** — the same stack you already use for
Django. The key differences from a typical Django deployment are:

- **FastAPI is ASGI, not WSGI.** You cannot use Gunicorn's default sync
  workers. The `uvicorn.workers.UvicornWorker` worker class is required, and
  Gunicorn acts purely as the process supervisor.

- **Single worker is mandatory.** ChromaDB's `PersistentClient` writes to
  SQLite files and is not safe to share across OS processes. Running multiple
  Gunicorn workers on the same data directory causes `database is locked`
  errors. One async worker handles high concurrency natively; embedding is
  CPU-bound and runs in FastAPI's built-in thread pool. If you need more
  throughput, run multiple machines behind a load balancer, each with their
  own data directory.

- **The embedding model must be loaded before the service accepts requests.**
  Loading `sentence-transformers/all-MiniLM-L6-v2` takes 10–30 seconds. This
  guide ensures the model is pre-downloaded and warmed up at startup rather
  than on the first live request.

---

## 1. System Preparation

### Install system packages

**Amazon Linux 2023 (dnf):**
```bash
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-pip python3.11-devel \
                    nginx git gcc
```

**Amazon Linux 2 (yum):**
```bash
sudo yum update -y
sudo amazon-linux-extras enable python3.8
sudo yum install -y python3.8 python3.8-pip python3.8-devel nginx git gcc
# Use python3.8 / pip3.8 wherever python3.11 / pip3.11 appears below
```

### Create the service user

```bash
sudo useradd --system --no-create-home --shell /sbin/nologin embeddingsvc
```

### Create directory structure

```bash
# Application code
sudo mkdir -p /srv/shared-embedding

# Persistent data — put this on an EBS volume if available
sudo mkdir -p /var/lib/shared-embedding/chroma_data
sudo mkdir -p /var/lib/shared-embedding/model-cache

# Log files
sudo mkdir -p /var/log/shared-embedding

# Set ownership
sudo chown -R embeddingsvc:embeddingsvc /srv/shared-embedding
sudo chown -R embeddingsvc:embeddingsvc /var/lib/shared-embedding
sudo chown -R embeddingsvc:embeddingsvc /var/log/shared-embedding
```

---

## 2. Deploy Application Code

### Copy files to the server

From your development machine:

```bash
rsync -av --exclude '__pycache__' --exclude '*.pyc' --exclude '.env' \
      --exclude 'chroma_data' --exclude 'apps.db' \
      ./ user@your-server:/srv/shared-embedding/
```

Or clone from your repository:

```bash
sudo -u embeddingsvc git clone https://your-repo-url /srv/shared-embedding
```

### Create and populate the virtual environment

```bash
sudo -u embeddingsvc python3.11 -m venv /srv/shared-embedding/venv
sudo -u embeddingsvc /srv/shared-embedding/venv/bin/pip install --upgrade pip
sudo -u embeddingsvc /srv/shared-embedding/venv/bin/pip install -r /srv/shared-embedding/requirements.txt
```

Gunicorn and the Uvicorn worker class are not in `requirements.txt` because
they are server-side only. Install them separately:

```bash
sudo -u embeddingsvc /srv/shared-embedding/venv/bin/pip install \
    "gunicorn>=21" "uvicorn[standard]>=0.29"
```

### Create the production `.env` file

```bash
sudo -u embeddingsvc tee /srv/shared-embedding/.env > /dev/null << 'EOF'
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=/var/lib/shared-embedding/chroma_data
DB_PATH=/var/lib/shared-embedding/apps.db
HF_HOME=/var/lib/shared-embedding/model-cache
EOF
sudo chmod 640 /srv/shared-embedding/.env
```

`HF_HOME` directs Hugging Face to cache the downloaded model under
`/var/lib/shared-embedding/model-cache` instead of the service user's home
directory. This keeps the model on your persistent EBS volume and makes the
path predictable.

---

## 3. Pre-download the Embedding Model

The model must be downloaded before the service starts for the first time.
This is a one-time step that takes 1–5 minutes depending on network speed.

```bash
sudo -u embeddingsvc bash -c '
    export HF_HOME=/var/lib/shared-embedding/model-cache
    /srv/shared-embedding/venv/bin/python -c "
from sentence_transformers import SentenceTransformer
print(\"Downloading model...\")
SentenceTransformer(\"all-MiniLM-L6-v2\")
print(\"Model ready.\")
"'
```

You only need to repeat this step when `EMBEDDING_MODEL` is changed in `.env`
(see [Changing the Embedding Model](#changing-the-embedding-model)).

---

## 4. Add Model Warmup to `main.py`

By default the embedding model loads lazily on the first request. In
production, add an explicit warmup call to the startup lifespan so the model
is loaded and cached before any connections are accepted.

Edit `/srv/shared-embedding/main.py` — replace the lifespan function:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Warm up the embedding model so the first real request is not delayed.
    # get_model() is lru_cache'd — this call populates the cache once.
    from app.embeddings import get_model
    import asyncio
    await asyncio.get_event_loop().run_in_executor(None, get_model)
    yield
```

---

## 5. Gunicorn Configuration

Create the Gunicorn config file:

```bash
sudo -u embeddingsvc tee /srv/shared-embedding/gunicorn.conf.py > /dev/null << 'EOF'
# ── Socket ────────────────────────────────────────────────────────────────
bind = "unix:/run/shared-embedding/gunicorn.sock"

# ── Workers ───────────────────────────────────────────────────────────────
# Single worker is intentional — see production_installation.md overview.
# Do not increase this without switching to a ChromaDB HTTP server first.
workers      = 1
worker_class = "uvicorn.workers.UvicornWorker"
threads      = 4   # concurrent sync-route execution within the single worker

# ── Timeouts ──────────────────────────────────────────────────────────────
timeout          = 120   # embedding large batches takes time
graceful_timeout = 30    # allow in-flight requests to finish on reload
keepalive        = 5

# ── Pre-loading ───────────────────────────────────────────────────────────
# Import the app in the master process before forking so startup code
# (including model warmup) runs once, not once per worker.
preload_app = True

# ── Worker recycling ──────────────────────────────────────────────────────
max_requests        = 1000
max_requests_jitter = 100

# ── Logging ───────────────────────────────────────────────────────────────
accesslog         = "/var/log/shared-embedding/access.log"
errorlog          = "/var/log/shared-embedding/error.log"
loglevel          = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sµs'
EOF
```

---

## 6. systemd Service

This service uses a single systemd unit (no separate `.socket` unit). Gunicorn
creates and owns the socket file itself, which is simpler and behaves
identically to your Django deployments.

### Allow Nginx to read the socket

```bash
sudo usermod -aG embeddingsvc nginx
```

### Create the service unit

```bash
sudo tee /etc/systemd/system/shared-embedding.service > /dev/null << 'EOF'
[Unit]
Description=Shared Embedding Service
After=network.target

[Service]
Type=notify
User=embeddingsvc
Group=embeddingsvc
WorkingDirectory=/srv/shared-embedding
EnvironmentFile=/srv/shared-embedding/.env

# Create /run/shared-embedding/ on start, remove it on stop.
# The socket file lives here.  Mode 0750 + nginx in embeddingsvc group
# allows Nginx to connect while keeping the socket private otherwise.
RuntimeDirectory=shared-embedding
RuntimeDirectoryMode=0750

# UMask 0007 makes the socket file 0660 (owner + group read/write).
UMask=0007

ExecStart=/srv/shared-embedding/venv/bin/gunicorn main:app -c gunicorn.conf.py

# Graceful reload: send SIGHUP to the master, which hot-restarts workers
# without closing the socket.  In-flight requests finish on old workers.
ExecReload=/bin/kill -s HUP $MAINPID

KillMode=mixed
TimeoutStopSec=30
Restart=on-failure
RestartSec=5s

# ── Security hardening ───────────────────────────────────────────────────
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
# Explicitly grant write access to the two persistent data paths.
ReadWritePaths=/var/lib/shared-embedding /var/log/shared-embedding

[Install]
WantedBy=multi-user.target
EOF
```

### Enable and verify

```bash
sudo systemctl daemon-reload
sudo systemctl enable shared-embedding
```

---

## 7. Nginx Configuration

Create a virtual host configuration:

```bash
sudo tee /etc/nginx/conf.d/shared-embedding.conf > /dev/null << 'EOF'
upstream shared_embedding_upstream {
    server unix:/run/shared-embedding/gunicorn.sock fail_timeout=0;
}

server {
    listen 80;
    server_name 127.0.0.1:8100;   # replace with your hostname or internal IP if needed

    # ── Access control ────────────────────────────────────────────────────
    # This service should only be reachable from your Django app servers.
    # Adjust to match your VPC / private subnet CIDR, then also add an EC2
    # security group rule restricting port 80 to the same source.
    # allow 10.0.0.0/8;
    # allow 172.16.0.0/12;
    # allow 192.168.0.0/16;
    # allow 127.0.0.1;
    deny all;

    # ── Request limits ────────────────────────────────────────────────────
    client_max_body_size 20M;   # accommodate large document batches

    # ── Proxy ─────────────────────────────────────────────────────────────
    location / {
        proxy_pass         http://shared_embedding_upstream;
        proxy_redirect     off;

        proxy_set_header   Host              $http_host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        proxy_connect_timeout  10s;
        proxy_read_timeout    120s;   # must be >= gunicorn timeout
        proxy_send_timeout    120s;
    }

    # ── Health check ──────────────────────────────────────────────────────
    # Suppress access log noise from load-balancer pings.
    location = / {
        proxy_pass  http://shared_embedding_upstream;
        access_log  off;
    }
}
EOF
```

Test and reload Nginx:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 8. First Start and Verification

```bash
# Start the service
sudo systemctl start shared-embedding

# Watch startup logs in real time (model loading takes ~15 seconds)
sudo journalctl -u shared-embedding -f
```

When ready you will see a line similar to:
```
[INFO] Uvicorn running on unix:/run/shared-embedding/gunicorn.sock
```

Verify the socket exists and is accessible:

```bash
ls -la /run/shared-embedding/gunicorn.sock
# Should show: srw-rw---- embeddingsvc embeddingsvc
```

Verify the service responds through Nginx:

```bash
curl -s http://localhost/
# Expected: {"status":"ok","service":"Shared Embedding Service","version":"1.0.0"}
```

Register your first app:

```bash
curl -s -X POST http://localhost/apps/register \
     -H "Content-Type: application/json" \
     -d '{"name": "pi-assist"}' | python3 -m json.tool
```

Save the `api_key` from the response — it is shown only once.

---

## 9. Managing the Service

### Status and logs

```bash
# Current status
sudo systemctl status shared-embedding

# Live log stream (Ctrl-C to exit)
sudo journalctl -u shared-embedding -f

# Last 100 lines
sudo journalctl -u shared-embedding -n 100

# Application-level access log
sudo tail -f /var/log/shared-embedding/access.log

# Application-level error log
sudo tail -f /var/log/shared-embedding/error.log
```

### Stop

```bash
sudo systemctl stop shared-embedding
```

Gunicorn waits up to `graceful_timeout` (30 s) for in-flight requests to finish
before killing workers. The Unix socket is removed when the service stops.

### Start

```bash
sudo systemctl start shared-embedding
```

Startup takes 15–30 seconds while the embedding model loads into memory. Nginx
will return `502 Bad Gateway` until the socket appears. If you need zero
downtime, use **reload** instead (see below).

### Reload (zero-downtime restart) — use for code changes

```bash
sudo systemctl reload shared-embedding
```

Sends `SIGHUP` to the Gunicorn master. The master spawns a new worker with
fresh code, waits for it to become healthy, then gracefully shuts down the old
worker. The Unix socket stays open throughout and Nginx sees no interruption.

Use `reload` after pulling updated code. It is not sufficient after changing
`requirements.txt` or `.env` — use a **restart** in those cases.

### Restart (brief downtime) — use for dependency or config changes

```bash
sudo systemctl restart shared-embedding
```

Stops then starts the service. The socket disappears for a few seconds; Nginx
will queue or briefly error on requests during that window. Restart when you
have updated Python packages or changed `.env` values.

### Enable / disable auto-start on boot

```bash
sudo systemctl enable shared-embedding    # enable (already done at install)
sudo systemctl disable shared-embedding   # disable
```

---

## 10. Updating the Service

### Standard update (code changes only)

```bash
# 1. Pull new code
cd /srv/shared-embedding
sudo -u embeddingsvc git pull

# 2. Verify the app imports cleanly before restarting
sudo -u embeddingsvc bash -c '
    export $(cat /srv/shared-embedding/.env | xargs)
    /srv/shared-embedding/venv/bin/python -c "from main import app; print(\"OK\")"
'

# 3. Graceful reload — zero downtime
sudo systemctl reload shared-embedding

# 4. Confirm it came back healthy
sudo systemctl status shared-embedding
curl -s http://localhost/ | python3 -m json.tool
```

### Update with new dependencies

```bash
# 1. Pull new code
cd /srv/shared-embedding
sudo -u embeddingsvc git pull

# 2. Install updated packages
sudo -u embeddingsvc /srv/shared-embedding/venv/bin/pip install \
    -r /srv/shared-embedding/requirements.txt

# 3. Restart (brief downtime — new packages are not visible to a reload)
sudo systemctl restart shared-embedding

sudo systemctl status shared-embedding
```

### Changing the Embedding Model

> **Warning:** Changing `EMBEDDING_MODEL` invalidates all existing embeddings
> in ChromaDB. Every Django app must re-ingest its documents after this change.
> Coordinate across all apps before proceeding.

```bash
# 1. Update EMBEDDING_MODEL in .env
sudo nano /srv/shared-embedding/.env

# 2. Pre-download the new model as the service user
sudo -u embeddingsvc bash -c '
    export HF_HOME=/var/lib/shared-embedding/model-cache
    /srv/shared-embedding/venv/bin/python -c "
from sentence_transformers import SentenceTransformer
import os
model = os.environ[\"EMBEDDING_MODEL\"]
print(f\"Downloading {model}...\")
SentenceTransformer(model)
print(\"Done.\")
"'

# 3. Back up and then wipe ChromaDB (old vectors are incompatible)
sudo systemctl stop shared-embedding
sudo cp -r /var/lib/shared-embedding/chroma_data \
           /var/lib/shared-embedding/chroma_data.bak-$(date +%F)
sudo -u embeddingsvc rm -rf /var/lib/shared-embedding/chroma_data/*

# 4. Restart — full restart required; preload_app caches the old model
sudo systemctl start shared-embedding

# 5. Re-run ingest scripts in all Django apps
```

---

## 11. Log Rotation

Create a logrotate config so log files do not grow unbounded:

```bash
sudo tee /etc/logrotate.d/shared-embedding > /dev/null << 'EOF'
/var/log/shared-embedding/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        # Signal Gunicorn to reopen log file handles after rotation
        systemctl reload shared-embedding > /dev/null 2>&1 || true
    endscript
}
EOF
```

---

## 12. Data Backup

The service has two persistent data stores. Back them up on a regular schedule
before any deployment.

### Back up before a deployment

```bash
BACKUP_DIR="/var/backups/shared-embedding/$(date +%F_%H%M%S)"
sudo mkdir -p "$BACKUP_DIR"

# App registry (SQLite)
sudo cp /var/lib/shared-embedding/apps.db "$BACKUP_DIR/"

# Vector store (ChromaDB)
# The service can be running during this copy — SQLite WAL mode makes it safe.
sudo cp -r /var/lib/shared-embedding/chroma_data "$BACKUP_DIR/"

echo "Backup written to $BACKUP_DIR"
```

### Automated nightly backup to S3

```bash
sudo tee /etc/cron.d/shared-embedding-backup > /dev/null << 'EOF'
0 2 * * * root \
  DEST=s3://your-bucket/shared-embedding/$(date +\%F) && \
  aws s3 cp /var/lib/shared-embedding/apps.db        "$DEST/apps.db" && \
  aws s3 sync /var/lib/shared-embedding/chroma_data  "$DEST/chroma_data/"
EOF
```

Replace `your-bucket` with your actual bucket name. The EC2 instance needs an
IAM role with `s3:PutObject` on that bucket.

---

## 13. EC2 Security Group

The service should never be publicly accessible. Configure your security group
to allow inbound port 80 **only** from the security group(s) attached to your
Django application servers — not from `0.0.0.0/0`.

| Type | Port | Source |
|---|---|---|
| HTTP | 80 | Security group of Django app servers |
| SSH | 22 | Your management IP only |

If the Shared Embedding Service and all Django apps run on the same EC2
instance, you do not need to open port 80 at all — Nginx communicates with
Gunicorn via the local Unix socket, and Django apps talk to Nginx on
`127.0.0.1`.

---

## 14. Troubleshooting

### Service fails to start

```bash
sudo journalctl -u shared-embedding -n 50 --no-pager
```

Common causes:
- **`Address already in use`** — a stale socket file from a previous crash.
  Run `sudo rm /run/shared-embedding/gunicorn.sock` then start again. (`RuntimeDirectory` in the unit file normally prevents this by cleaning up on stop.)
- **`ModuleNotFoundError`** — a package is missing from the venv. Run
  `pip install -r requirements.txt` as the `embeddingsvc` user.
- **Model download fails at startup** — the model was not pre-downloaded.
  Follow the pre-download step in [Section 3](#3-pre-download-the-embedding-model).

### 502 Bad Gateway from Nginx

```bash
# Is the service running?
sudo systemctl status shared-embedding

# Does the socket exist?
ls -la /run/shared-embedding/gunicorn.sock

# Can Nginx read the socket? (nginx must be in embeddingsvc group)
groups nginx
```

If nginx is not in the `embeddingsvc` group, add it and restart both services:

```bash
sudo usermod -aG embeddingsvc nginx
sudo systemctl restart nginx
sudo systemctl restart shared-embedding
```

### Request timeouts

Embedding large batches can approach the 120-second timeout. Increase
`timeout` and `proxy_read_timeout` / `proxy_send_timeout` in `gunicorn.conf.py`
and `shared-embedding.conf` respectively, then reload both services.

### ChromaDB `database is locked`

This means more than one process is writing to the same SQLite file. Confirm
`workers = 1` is set in `gunicorn.conf.py` and that no other process (e.g. a
management command or cron job) is accessing the ChromaDB directory while the
service is running.

### Check collection counts

```bash
API_KEY="ses-..."   # replace with a registered key
curl -s http://localhost/collections \
     -H "X-API-Key: $API_KEY" | python3 -m json.tool
```
