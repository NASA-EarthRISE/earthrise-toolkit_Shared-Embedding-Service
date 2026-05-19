# Production Installation Guide

## Overview

The Shared Embedding Service runs on **Gunicorn with Uvicorn workers**, fronted
by **Nginx**, managed by **systemd** — the same stack used for all EarthRISE
Django apps. Configuration follows the same conventions: `nginx` user, Anaconda
environment, inline Gunicorn flags in the service file, and the shared
`/earthrise_apps/socks/` directory.

There are two differences from the Django service files you already have:

- **Worker class must be `uvicorn.workers.UvicornWorker`, not `gevent`.**
  FastAPI is ASGI, not WSGI. Gunicorn's `gevent` worker runs a WSGI app;
  it will not work here. `UvicornWorker` is the ASGI equivalent and behaves
  the same way from a process-management perspective.

- **Workers must be set to `1`.** ChromaDB's `PersistentClient` writes to
  SQLite and is not safe to share across OS processes. Multiple workers on the
  same data directory will produce `database is locked` errors. A single async
  worker handles high concurrency natively; CPU-bound embedding runs in
  FastAPI's built-in thread pool. Scale out by adding machines, not workers.

---

## Paths Reference

| What | Path |
|---|---|
| Application code | `/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/` |
| Conda environment | `/opt/anaconda3/envs/shared_embedding/` |
| Unix socket | `/earthrise_apps/socks/shared_embedding.sock` |
| Persistent data | `/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/` |
| Log files | `/var/log/shared-embedding/` |
| systemd unit | `/etc/systemd/system/shared-embedding.service` |
| Nginx config | `/etc/nginx/conf.d/shared-embedding.conf` |

---

## 1. Create Directories

```bash
# Persistent data (chroma vector store, app registry, model cache)
sudo mkdir -p /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/chroma_data
sudo mkdir -p /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/model-cache

# Log files
sudo mkdir -p /var/log/shared-embedding

# Set ownership — nginx user runs the service
sudo chown -R nginx:nginx /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data
sudo chown -R nginx:nginx /var/log/shared-embedding
```

---

## 2. Deploy Application Code

```bash
cd /earthrise_apps
sudo git clone https://github.com/NASA-EarthRISE/earthrise-toolkit_shared-embedding-service \
    earthrise-toolkit_Shared-Embedding-Service
sudo chown -R nginx:nginx /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service
```

---

## 3. Create the Conda Environment

```bash
sudo /opt/anaconda3/bin/conda create -n shared_embedding python=3.13 -y
```

Install application dependencies:

```bash
sudo /opt/anaconda3/envs/shared_embedding/bin/pip install \
    -r /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/requirements.txt
```

Install the Gunicorn and Uvicorn server packages (server-side only, not in
`requirements.txt`):

```bash
sudo /opt/anaconda3/envs/shared_embedding/bin/pip install \
    "gunicorn>=21" "uvicorn[standard]>=0.29"
```

---

## 4. Create the Production `.env` File

```bash
sudo -u nginx tee /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/.env > /dev/null << 'EOF'
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/chroma_data
DB_PATH=/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/apps.db
HF_HOME=/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/model-cache
EOF
sudo chmod 640 /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/.env
```

`HF_HOME` directs Hugging Face to store the downloaded model in the app's
`data/` directory rather than a user home. This keeps the model on the same
EBS volume as the vector store and makes the path predictable across restarts.

---

## 5. Pre-download the Embedding Model

The model must be present on disk before the service starts. This one-time step
takes 1–5 minutes depending on network speed.

```bash
sudo -u nginx bash -c '
    export HF_HOME=/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/model-cache
    /opt/anaconda3/envs/shared_embedding/bin/python -c "
from sentence_transformers import SentenceTransformer
print(\"Downloading model...\")
SentenceTransformer(\"all-MiniLM-L6-v2\")
print(\"Model ready.\")
"'
```

Repeat this step only when `EMBEDDING_MODEL` is changed in `.env`
(see [Changing the Embedding Model](#changing-the-embedding-model)).

---

## 6. Add Model Warmup to `main.py`

By default the embedding model loads lazily on the first request. Add an
explicit warmup call to the FastAPI startup lifespan so the model is loaded and
cached before any connections are accepted. Edit
`/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/main.py` and
replace the existing lifespan function:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Pre-load the embedding model so the first real request is not delayed.
    # get_model() is lru_cache'd — this call populates the cache once at startup.
    from app.embeddings import get_model
    import asyncio
    await asyncio.get_event_loop().run_in_executor(None, get_model)
    yield
```

---

## 7. systemd Service

```bash
sudo tee /etc/systemd/system/shared-embedding.service > /dev/null << 'EOF'
[Unit]
Description=Shared Embedding Service daemon
After=network.target

[Service]
User=nginx
Group=nginx
WorkingDirectory=/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service
EnvironmentFile=/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/.env
ExecStart=/opt/anaconda3/envs/shared_embedding/bin/gunicorn \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --backlog 100 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --bind unix:/earthrise_apps/socks/shared_embedding.sock \
    --access-logfile /var/log/shared-embedding/access.log \
    --error-logfile /var/log/shared-embedding/error.log \
    --log-level info \
    --pythonpath '/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service,/opt/anaconda3/envs/shared_embedding/lib/python3.13/site-packages' \
    main:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=30
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
```

Flag notes:

| Flag | Value | Reason |
|---|---|---|
| `--worker-class` | `uvicorn.workers.UvicornWorker` | Required for ASGI — replaces `gevent` |
| `--workers` | `1` | ChromaDB single-writer constraint — do not increase |
| `--threads` | `4` | Concurrent sync-route execution within the single worker |
| `--timeout` | `120` | Embedding large batches takes time |
| `--preload` | *(flag)* | Loads app and model in master process before forking; keeps startup fast |
| `--max-requests` | `1000` | Recycles the worker periodically to guard against memory growth |

Enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable shared-embedding
```

---

## 8. Nginx Configuration

```bash
sudo tee /etc/nginx/conf.d/shared-embedding.conf > /dev/null << 'EOF'
upstream shared_embedding_upstream {
    server unix:/earthrise_apps/socks/shared_embedding.sock fail_timeout=0;
}

server {
    listen 80;
    server_name 127.0.0.1;   # update to your Route 53 internal hostname if on a dedicated host

    # ── Access control ────────────────────────────────────────────────────
    # Uncomment and adjust to your VPC CIDR once the service is on its own host.
    # Also add a matching EC2 security group rule for port 80.
    # allow 10.0.0.0/8;
    # allow 172.16.0.0/12;
    # allow 192.168.0.0/16;
    allow 127.0.0.1;
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
        proxy_read_timeout    120s;   # must be >= gunicorn --timeout
        proxy_send_timeout    120s;
    }

    # ── Health check ──────────────────────────────────────────────────────
    location = / {
        proxy_pass  http://shared_embedding_upstream;
        access_log  off;
    }
}
EOF
```

Test and reload Nginx:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

---

## 9. First Start and Verification

```bash
# Start the service
sudo systemctl start shared-embedding

# Watch startup logs — model loading takes ~15 seconds
sudo journalctl -u shared-embedding -f
```

When ready you will see a line similar to:
```
[INFO] Uvicorn running on unix:/earthrise_apps/socks/shared_embedding.sock
```

Verify the socket exists:

```bash
ls -la /earthrise_apps/socks/shared_embedding.sock
# Should show: srw-rw-rw- nginx nginx  (or srw-r--r--)
```

Verify the service responds through Nginx:

```bash
curl -s http://127.0.0.1/
# Expected: {"status":"ok","service":"Shared Embedding Service","version":"1.0.0"}
```

Register your first app:

```bash
curl -s -X POST http://127.0.0.1/apps/register \
     -H "Content-Type: application/json" \
     -d '{"name": "pi-assist"}' | python3 -m json.tool
```

Save the `api_key` from the response — it is shown only once.

---

## 10. Managing the Service

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

Gunicorn waits up to 30 seconds (`TimeoutStopSec`) for in-flight requests to
finish before killing the worker. The socket file is removed when the process
exits.

### Start

```bash
sudo systemctl start shared-embedding
```

Startup takes 15–30 seconds while the embedding model loads into memory
(`--preload` ensures this happens before the socket is created). Nginx will
return `502 Bad Gateway` until the socket appears.

### Reload — zero-downtime restart for code changes

```bash
sudo systemctl reload shared-embedding
```

Sends `SIGHUP` to the Gunicorn master. The master spawns a new worker with
fresh code, waits for it to become healthy, then gracefully shuts down the old
worker. The Unix socket stays open and Nginx sees no interruption.

Use `reload` after pulling updated application code. It is **not** sufficient
after changing `requirements.txt`, `.env`, or `--preload`-cached state — use a
full **restart** in those cases.

### Restart — required for dependency or config changes

```bash
sudo systemctl restart shared-embedding
```

Stops then starts the service. The socket disappears for a few seconds during
the model load; Nginx will return `502` during that window.

### Enable / disable auto-start on boot

```bash
sudo systemctl enable shared-embedding    # already done at install
sudo systemctl disable shared-embedding
```

---

## 11. Updating the Service

### Code changes only

```bash
# 1. Pull new code
cd /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service
sudo git pull

# 2. Verify the app imports cleanly before restarting
sudo -u nginx bash -c '
    export $(grep -v "^#" /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/.env | xargs)
    /opt/anaconda3/envs/shared_embedding/bin/python -c "from main import app; print(\"OK\")"
'

# 3. Graceful reload — zero downtime
sudo systemctl reload shared-embedding

# 4. Confirm healthy
sudo systemctl status shared-embedding
curl -s http://127.0.0.1/ | python3 -m json.tool
```

### New or updated dependencies

```bash
# 1. Pull new code
cd /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service
sudo git pull

# 2. Install updated packages into the conda env
sudo /opt/anaconda3/envs/shared_embedding/bin/pip install \
    -r /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/requirements.txt

# 3. Full restart — new packages are not visible to a reload
sudo systemctl restart shared-embedding
sudo systemctl status shared-embedding
```

### Changing the Embedding Model

> **Warning:** Changing `EMBEDDING_MODEL` invalidates all existing embeddings
> in ChromaDB. Every Django app must re-ingest its documents after this change.
> Coordinate across all apps before proceeding.

```bash
# 1. Update EMBEDDING_MODEL in .env
sudo nano /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/.env

# 2. Pre-download the new model
sudo -u nginx bash -c '
    export HF_HOME=/earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/model-cache
    /opt/anaconda3/envs/shared_embedding/bin/python -c "
from sentence_transformers import SentenceTransformer
import os; model = os.environ[\"EMBEDDING_MODEL\"]
print(f\"Downloading {model}...\")
SentenceTransformer(model)
print(\"Done.\")
"'

# 3. Back up then wipe ChromaDB (old vectors are incompatible with the new model)
sudo systemctl stop shared-embedding
sudo cp -r /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/chroma_data \
           /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/chroma_data.bak-$(date +%F)
sudo rm -rf /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/chroma_data/*
sudo chown -R nginx:nginx /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data

# 4. Full restart required — --preload cached the old model in memory
sudo systemctl start shared-embedding

# 5. Re-run ingest scripts in all Django apps
```

---

## 12. Log Rotation

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
        systemctl reload shared-embedding > /dev/null 2>&1 || true
    endscript
}
EOF
```

---

## 13. Data Backup

The service has two persistent data stores. Back up before every deployment.

### Manual backup

```bash
BACKUP_DIR="/var/backups/shared-embedding/$(date +%F_%H%M%S)"
sudo mkdir -p "$BACKUP_DIR"

# App registry (SQLite)
sudo cp /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/apps.db \
        "$BACKUP_DIR/"

# Vector store (ChromaDB) — safe to copy while the service is running
sudo cp -r /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/chroma_data \
           "$BACKUP_DIR/"

echo "Backup written to $BACKUP_DIR"
```

### Automated nightly backup to S3

```bash
sudo tee /etc/cron.d/shared-embedding-backup > /dev/null << 'EOF'
0 2 * * * root \
  DEST=s3://your-bucket/shared-embedding/$(date +\%F) && \
  aws s3 cp /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/apps.db \
            "$DEST/apps.db" && \
  aws s3 sync /earthrise_apps/earthrise-toolkit_Shared-Embedding-Service/data/chroma_data \
              "$DEST/chroma_data/"
EOF
```

Replace `your-bucket` with your actual S3 bucket name. The EC2 instance needs
an IAM role with `s3:PutObject` permission on that bucket.

---

## 14. EC2 Security Group

The service should never be publicly accessible. Configure your security group
to allow inbound port 80 **only** from the security group(s) attached to your
Django application servers.

| Type | Port | Source |
|---|---|---|
| HTTP | 80 | Security group of Django app servers |
| SSH | 22 | Your management IP only |

If the Shared Embedding Service and all Django apps run on the same EC2
instance, no inbound port 80 rule is needed at all — Django apps communicate
with the service via `http://127.0.0.1` and Nginx routes over the local socket.

---

## 15. Troubleshooting

### Service fails to start

```bash
sudo journalctl -u shared-embedding -n 50 --no-pager
```

Common causes:

- **`Address already in use`** — stale socket file from a previous crash.
  Remove it and start again:
  ```bash
  sudo rm /earthrise_apps/socks/shared_embedding.sock
  sudo systemctl start shared-embedding
  ```
- **`ModuleNotFoundError`** — a package is missing from the conda env. Run
  `pip install -r requirements.txt` against the conda env's pip.
- **Model not found at startup** — the model was not pre-downloaded. Re-run
  the pre-download step in [Section 5](#5-pre-download-the-embedding-model).

### 502 Bad Gateway from Nginx

```bash
# Is the service running?
sudo systemctl status shared-embedding

# Does the socket exist?
ls -la /earthrise_apps/socks/shared_embedding.sock

# Do the Nginx config and upstream path match?
grep -r shared_embedding /etc/nginx/conf.d/
```

### Request timeouts

Embedding large batches can approach the 120-second limit. Increase both
`--timeout` in the service file and `proxy_read_timeout` / `proxy_send_timeout`
in `shared-embedding.conf`, then restart the service and reload Nginx.

### ChromaDB `database is locked`

More than one process is writing to the same SQLite file. Confirm `--workers 1`
is set in the service file. If a management script or cron job also touches the
ChromaDB directory, stop the service first, run the script, then restart.

### Check collection counts

```bash
API_KEY="ses-..."
curl -s http://127.0.0.1/collections \
     -H "X-API-Key: $API_KEY" | python3 -m json.tool
```
