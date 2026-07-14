# Smart Road & Flyover Damage Monitoring System
## Complete Deployment Guide (Beginner-Friendly)

This guide takes you from a fresh folder to a running, production-ready
Streamlit application — locally in VS Code and then deployed to the cloud.

---

## 1. Project Structure

Place all files in a single project folder like this:

```
smart-road-damage-monitoring/
├── app.py                        # ⭐ MAIN ENTRY POINT — run this
├── database.py                   # SQLite persistence layer
├── pdf_report_service.py         # PDF Inspection Report generator
├── config.py                     # Central configuration
├── detection_service.py          # YOLOv8 inference engine
├── image_utils.py
├── video_utils.py
├── analytics_service.py
├── health_service.py
├── risk_service.py
├── emergency_service.py
├── repair_cost_service.py
├── life_prediction_service.py
├── simulation_service.py
├── timeline_service.py
├── recommendation_service.py
├── notification_service.py
├── complaint_service.py
├── explainability_service.py
├── score_utils.py
├── requirements.txt
├── models/
│   ├── best.pt                   # your trained YOLOv8 weights (optional)
│   └── ml/                       # optional trained XGBoost models
└── data/                         # auto-created at runtime (db, exports, logs)
```

> **Note:** `dashboard.py` (an earlier draft) is superseded by `app.py`.
> `app.py` contains everything `dashboard.py` had, plus the PDF Report page.
> Run `app.py`, not `dashboard.py`.

---

## 2. Prerequisites

| Requirement | Version | Check with |
|---|---|---|
| Python | 3.10 – 3.11 (3.12 works but ultralytics may lag) | `python --version` |
| pip | latest | `pip --version` |
| VS Code | latest | — |
| Git (optional) | latest | `git --version` |
| ~2 GB free disk | for model weights + libraries | — |

On Windows, also install the "Desktop development with C++" workload if
`opencv-python` fails to build (rare — prebuilt wheels usually work).

---

## 3. Installation Steps

### Step 1 — Create the project folder and virtual environment

```bash
mkdir smart-road-damage-monitoring
cd smart-road-damage-monitoring

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt once activated.

### Step 2 — Copy in all project files

Copy `app.py`, `database.py`, `pdf_report_service.py`, `config.py`, and every
`*_service.py` / `*_utils.py` file from this project into the folder above.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` (already provided) includes: `streamlit`, `ultralytics`,
`opencv-python`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`,
`plotly`, `reportlab`, `Pillow`, `imageio`, `imageio-ffmpeg`, `SQLAlchemy`,
`python-dotenv`, `pydantic`, `tqdm`, `joblib`.

If you don't have a trained YOLOv8 model yet, that's fine — skip Step 4.
The app detects the missing weights automatically and switches every page
to a **manual detection entry** mode so you can still exercise the entire
scoring pipeline (health, risk, emergency, cost, RUL, simulation,
recommendations, PDF report) end-to-end.

### Step 4 — (Optional) Add your trained YOLOv8 weights

```
models/best.pt
```

If you don't have one yet, train with Ultralytics:

```bash
yolo detect train data=your_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
# then copy runs/detect/train/weights/best.pt -> models/best.pt
```

### Step 5 — Initialize the database (optional — auto-runs on first launch)

```bash
python database.py
```

This creates `data/database/road_monitoring.db` with all 5 tables
(`detections`, `predictions`, `reports`, `notifications`, `complaints`)
plus a `roads` lookup table.

---

## 4. VS Code Execution Guide

1. Open the project folder in VS Code: `File → Open Folder…`
2. Install the **Python extension** (Microsoft) if you don't have it.
3. Select the interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter` →
   choose the `venv` you created in Step 1.
4. Open a new integrated terminal: `` Ctrl+` `` (it should auto-activate
   `venv` — confirm you see `(venv)` in the prompt).
5. Run the app:

   ```bash
   streamlit run app.py
   ```

6. VS Code/Streamlit will print a local URL, typically:

   ```
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

   Ctrl+Click the Local URL (or open it in your browser) to view the app.

7. **Live reload while developing:** Streamlit auto-detects file changes.
   When it shows "Source file changed" in the browser, click **Rerun** (or
   enable "Always rerun" in the top-right menu) to see edits instantly.

8. **Debugging in VS Code:** create `.vscode/launch.json`:

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Streamlit: app.py",
         "type": "python",
         "request": "launch",
         "module": "streamlit",
         "args": ["run", "app.py"],
         "justMyCode": true
       }
     ]
   }
   ```

   Then press `F5` to run with breakpoints enabled.

---

## 5. Using the App (Quick Walkthrough)

1. **Sidebar** — pick or create a Road/Flyover asset, set its age and
   traffic load.
2. **📤 Upload & Detect** — upload an image or video (or use the Manual
   Detections tab if you have no trained model yet). Detections are
   saved to SQLite immediately.
3. **🧠 Decision Intelligence** — click **Run Full Scoring Pipeline** to
   compute Health Score → Risk Score → Emergency Index → Repair Cost →
   Remaining Useful Life → Recommendations, all in one call. Optionally
   send a notification or generate a municipality complaint here.
4. **📈 Damage Growth Simulation** — view optimistic/expected/pessimistic
   forecasts for the next N months.
5. **🔍 SHAP / Explainability** — see exactly which factors drove the
   Emergency Index and Health Score.
6. **📄 PDF Report** — click **Generate PDF Report** to produce the full
   inspection report (date, image, damages, scores, recommendations,
   complaint letter) and download it.
7. **🕘 Detection History** / **📨 Notifications & Complaints** /
   **📊 Analytics Dashboard** — review everything ever recorded, with
   charts (damage distribution pie, severity bar, risk trend, repair
   cost, growth forecast).

---

## 6. Deployment Instructions

### Option A — Streamlit Community Cloud (fastest, free tier)

1. Push your project to a **public or private GitHub repository**
   (include `requirements.txt`; do **not** commit `data/` or large model
   weights — add them to `.gitignore` and use Git LFS or external storage
   for `models/yolo/best.pt` if it's large).
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo, branch, and set **Main file path** to `app.py`.
4. Under **Advanced settings**, set Python version to match your local
   venv (3.10/3.11).
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL.
6. **Persistence caveat:** Streamlit Cloud's filesystem is ephemeral on
   redeploy/restart. For a real production deployment, point
   `config.DATABASE_PATH` at a persistent volume or migrate `database.py`
   to a hosted Postgres/MySQL instance (the SQL is simple enough to port).

### Option B — Docker (recommended for production)

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t road-damage-monitor .
docker run -p 8501:8501 -v $(pwd)/data:/app/data road-damage-monitor
```

The `-v $(pwd)/data:/app/data` mounts a host folder so your SQLite
database and generated reports survive container restarts.

Push to any container registry (Docker Hub, ECR, GCR) and deploy on:
- **AWS**: ECS Fargate, Elastic Beanstalk, or EC2 + Docker
- **GCP**: Cloud Run (set concurrency=1 if using in-memory model caching)
- **Azure**: App Service (container) or Container Apps
- **Render / Railway / Fly.io**: point at the Dockerfile, attach a
  persistent volume for `/app/data`

### Option C — Traditional VM / on-prem server

```bash
# On the server
git clone <your-repo>
cd smart-road-damage-monitoring
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run behind a process manager so it survives reboots/crashes
pip install --break-system-packages supervisor   # or use systemd
```

Example `systemd` service (`/etc/systemd/system/road-monitor.service`):

```ini
[Unit]
Description=Smart Road Damage Monitoring Streamlit App
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/smart-road-damage-monitoring
Environment="APP_ENV=production"
ExecStart=/opt/smart-road-damage-monitoring/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now road-monitor
```

Put **Nginx** in front for TLS + a real domain:

```nginx
server {
    listen 443 ssl;
    server_name roadmonitor.yourcity.gov;

    ssl_certificate     /etc/letsencrypt/live/roadmonitor.yourcity.gov/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/roadmonitor.yourcity.gov/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 7. Production Hardening Checklist

- [ ] Set `APP_ENV=production` (disables `DEBUG` in `config.py`).
- [ ] Move `DATABASE_PATH` to a persistent, backed-up volume (or migrate
      to Postgres for multi-instance deployments).
- [ ] Replace `notification_service.py`'s simulated email/SMS/push
      adapters with real providers (SMTP/SES for email, Twilio for SMS,
      FCM/APNs for push) — every adapter shares one `{channel, status,
      timestamp}` shape so this is a drop-in swap.
- [ ] Calibrate `repair_cost_service.DEFAULT_PIXELS_PER_METER` per your
      actual camera mount (height/angle/lens) — the shipped value is a
      placeholder.
- [ ] Put real trained models at `models/yolo/best.pt` and (optionally)
      `models/ml/priority_xgb_model.json` / `rul_xgb_model.json` /
      `repair_cost_xgb_model.json` for ML-mode scoring + true SHAP.
- [ ] Add authentication in front of the app (Streamlit Cloud's built-in
      viewer auth, or an Nginx `auth_basic` / OAuth2 proxy) before
      exposing this to the public internet — civic complaint data and
      cost estimates are sensitive.
- [ ] Enable HTTPS (Let's Encrypt via certbot, or your cloud provider's
      managed TLS).
- [ ] Set up log rotation for `logs/app.log`, `logs/detection.log`,
      `logs/error.log` (e.g. `logrotate` on Linux).
- [ ] Add automated backups of `data/database/road_monitoring.db` and
      `data/exports/reports/`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'X'` | Activate your venv, then `pip install -r requirements.txt` again. |
| `FileNotFoundError: YOLOv8 weights not found` | Expected if you haven't trained a model — use the Manual Detections tab, or add `models/yolo/best.pt`. |
| Blank/black boxes in the PDF report | Don't use Unicode sub/superscript characters in custom edits to `pdf_report_service.py` — use ReportLab's `<sub>`/`<super>` tags instead. |
| Video processing is slow | Lower `FRAME_SAMPLE_RATE_FPS` in `config.py`, or trim video length below `MAX_VIDEO_DURATION_SECONDS`. |
| `streamlit: command not found` | Your venv isn't activated, or Streamlit didn't install — re-run `pip install streamlit`. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| SQLite "database is locked" under concurrent users | Fine for demo/single-user use; for multi-user production, migrate to Postgres (schema in `database.py` is standard SQL and ports directly). |

---

## 9. Support Matrix Summary

| Component | Library | Fallback if unavailable |
|---|---|---|
| Detection | ultralytics YOLOv8 | Manual detection entry form |
| Priority/RUL/Repair-cost ML | XGBoost | Heuristic formulas (always available) |
| Explainability | SHAP (`shap.TreeExplainer`) | Exact rule-based weighted-sum decomposition |
| PDF Reports | ReportLab | N/A (always available, no ML dependency) |
| Charts | Plotly | N/A (always available) |
| Database | SQLite (`database.py`) | N/A — swap connection string for Postgres if scaling |

You now have a complete, beginner-friendly path from `git clone` to a
live, publicly accessible Smart Road & Flyover Damage Monitoring system.
