# AI-Powered Smart Road & Flyover Damage Monitoring and Decision Intelligence System
### System Architecture Document

---

## 1. Complete Folder Structure

```
smart-road-damage-monitoring/
├── app/                                # Streamlit presentation layer
│   ├── main.py                         # App entry point / router
│   ├── pages/
│   │   ├── 1_Dashboard.py              # KPI overview, map, alerts
│   │   ├── 2_Upload_Inspection.py      # Upload images/videos, trigger pipeline
│   │   ├── 3_Detection_Results.py      # Annotated media viewer
│   │   ├── 4_Severity_Analytics.py     # Severity distributions, trends
│   │   ├── 5_Maintenance_Decision.py   # XGBoost priority + SHAP explanations
│   │   ├── 6_Reports.py                # Generate/download PDF reports
│   │   └── 7_Admin_Settings.py         # Thresholds, users, model config
│   └── components/
│       ├── sidebar.py
│       ├── map_view.py                 # Geo-plot of road/flyover segments
│       ├── charts.py                   # Reusable Plotly components
│       └── alerts_widget.py
│
├── core/                                # Core business logic (framework-agnostic)
│   ├── detection/
│   │   ├── yolo_inference.py           # Loads best.pt, runs inference
│   │   ├── video_processor.py          # Frame extraction, sampling, tracking
│   │   ├── image_processor.py          # Single-image inference pipeline
│   │   └── postprocessing.py           # NMS cleanup, bbox → damage metrics
│   ├── severity/
│   │   ├── feature_extraction.py       # Area %, crack density, bbox geometry
│   │   ├── classifier.py               # Scikit-learn severity classifier
│   │   └── severity_engine.py          # Orchestrates severity scoring
│   ├── decision_intelligence/
│   │   ├── priority_model.py           # XGBoost maintenance-priority model
│   │   ├── risk_scoring.py             # Composite risk index computation
│   │   ├── explainability.py           # SHAP value generation
│   │   └── maintenance_planner.py      # Converts scores → action recommendations
│   ├── reporting/
│   │   ├── pdf_report_generator.py     # ReportLab report builder
│   │   ├── dashboard_charts.py         # Plotly figure factories
│   │   └── templates/                  # Report layout templates/branding assets
│   └── utils/
│       ├── geo_utils.py
│       ├── image_utils.py
│       ├── video_utils.py
│       └── validators.py               # Input/file/schema validation
│
├── database/
│   ├── db_manager.py                   # SQLite connection/session handling
│   ├── models.py                       # ORM-style table definitions
│   ├── schema.sql                      # DDL (see Section 6)
│   └── migrations/                     # Versioned schema migrations
│
├── models/
│   ├── yolo/
│   │   └── best.pt                     # Provided trained YOLOv8 weights
│   └── ml/
│       ├── severity_classifier.pkl     # Scikit-learn model artifact
│       ├── priority_xgb_model.json     # XGBoost model artifact
│       └── scaler.pkl                  # Feature scaler for ML inputs
│
├── data/
│   ├── raw/
│   │   ├── images/                     # Provided image dataset
│   │   └── videos/                     # Provided video dataset
│   ├── processed/
│   │   ├── annotated_images/
│   │   ├── annotated_videos/
│   │   └── frames/                     # Extracted video frames
│   ├── training/
│   │   ├── images/
│   │   └── labels/                     # YOLO-format labels (if retraining)
│   └── exports/
│       ├── reports/                    # Generated PDFs
│       └── csv_exports/                # Tabular exports for offline analysis
│
├── training/                            # Offline model development (not runtime)
│   ├── train_yolo.py                   # Optional fine-tuning entry point
│   ├── train_severity_model.py
│   ├── train_priority_model.py
│   └── evaluation/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── model_evaluation.ipynb
│   └── shap_analysis.ipynb
│
├── logs/
│   ├── app.log
│   ├── detection.log
│   └── error.log
│
├── tests/
│   ├── test_detection.py
│   ├── test_severity.py
│   ├── test_decision_engine.py
│   └── test_db.py
│
├── config.py                           # Central configuration
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 2. System Architecture (Layered View)

```
┌───────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER (Streamlit)                 │
│  Dashboard | Upload | Detection Viewer | Analytics | Decisions | Rpts │
└───────────────────────────────────────────────────────────────────────┘
                                   │  calls
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     APPLICATION / ORCHESTRATION LAYER                 │
│   Session control, pipeline sequencing, request validation, caching   │
└───────────────────────────────────────────────────────────────────────┘
        │                 │                  │                  │
        ▼                 ▼                  ▼                  ▼
┌───────────────┐ ┌───────────────┐ ┌──────────────────┐ ┌───────────────┐
│  DETECTION    │ │  SEVERITY     │ │  DECISION         │ │  REPORTING    │
│  LAYER        │ │  ANALYTICS    │ │  INTELLIGENCE     │ │  LAYER        │
│  YOLOv8       │ │  LAYER        │ │  LAYER            │ │  ReportLab    │
│  OpenCV       │ │  Scikit-learn │ │  XGBoost + SHAP   │ │  Plotly       │
└───────────────┘ └───────────────┘ └──────────────────┘ └───────────────┘
        │                 │                  │                  │
        └─────────────────┴────────┬─────────┴──────────────────┘
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE LAYER (SQLite)                    │
│  roads | sessions | media | detections | severity | predictions       │
│  recommendations | reports | alerts | users                           │
└───────────────────────────────────────────────────────────────────────┘
```

**Layer responsibilities:**
- **Presentation Layer** — Streamlit multipage app; pure UI, no business logic.
- **Application/Orchestration Layer** — Coordinates the pipeline: ingestion → detection → severity → decision → report; enforces validation and sequencing; caches heavy objects (model instances) via `st.cache_resource`.
- **Detection Layer** — YOLOv8 inference on images/video frames; OpenCV for frame extraction, resizing, annotation drawing.
- **Severity Analytics Layer** — Converts raw detections into quantified severity using geometric/statistical features and a Scikit-learn classifier.
- **Decision Intelligence Layer** — XGBoost model ranks maintenance priority/risk; SHAP explains each prediction for transparency to engineers/authorities.
- **Reporting Layer** — Plotly for interactive in-app visuals; ReportLab for exportable PDF inspection/decision reports.
- **Persistence Layer** — SQLite as the single source of truth for all entities across sessions.

---

## 3. Data Flow Diagram

```
 [Image/Video Dataset]         [Live Upload via Streamlit]
          │                              │
          └───────────────┬──────────────┘
                           ▼
                 ┌───────────────────┐
                 │  Media Ingestion  │  (validate, store, register in DB)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Preprocessing      │  (OpenCV: resize, denoise,
                 │ (image/video)      │   frame extraction @ N fps)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │  YOLOv8 Inference  │  (best.pt → bboxes, class,
                 │  (detection.core)  │   confidence per frame/image)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Postprocessing     │  (NMS, dedup across frames,
                 │                    │   damage-area computation)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Feature Extraction │  (crack length/area %, density,
                 │                    │   count per segment, confidence)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Severity Engine    │  (Scikit-learn classifier →
                 │ (Scikit-learn)     │   Low/Medium/High/Critical)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Risk & Priority    │  (XGBoost → maintenance priority
                 │ Model (XGBoost)    │   score / est. remaining life)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Explainability     │  (SHAP → feature contribution
                 │ (SHAP)             │   per prediction)
                 └───────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Maintenance        │  (rule + model hybrid →
                 │ Planner            │   recommended action, urgency)
                 └───────────────────┘
                           ▼
        ┌──────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐                 ┌───────────────────┐
│  SQLite Storage    │◄───────────────│  Alerts Generator  │
│  (all entities)    │                 │  (critical damage)│
└───────────────────┘                 └───────────────────┘
        ▼
┌───────────────────┐        ┌───────────────────┐
│ Streamlit Dashboard│        │  PDF Report        │
│ (Plotly visuals)   │        │  (ReportLab)        │
└───────────────────┘        └───────────────────┘
```

---

## 4. Complete Workflow

1. **Ingestion** — User uploads an image/video via Streamlit, or a batch job points at the existing dataset folders. File is validated (type/size), copied to `data/raw/`, and a `media_files` row is created linked to a `inspection_sessions` row.
2. **Preprocessing** — OpenCV extracts frames from video at a configurable sampling rate (`FRAME_SAMPLE_RATE`), or normalizes single images (resize, color correction).
3. **Detection** — Each frame/image is passed through the loaded YOLOv8 (`best.pt`) model. Bounding boxes, class labels (crack, pothole, spalling, joint failure, etc.), and confidence scores are produced.
4. **Postprocessing** — Duplicate detections across adjacent frames are merged/tracked; damage pixel-area and bounding-box geometry are computed; annotated media saved to `data/processed/`.
5. **Feature Extraction** — Aggregate per-segment features: damage count, average confidence, total damaged area %, crack density, damage type mix.
6. **Severity Classification** — A Scikit-learn model (trained offline on labeled severity outcomes, or rule-calibrated initially) assigns a severity class and continuous severity score per detection/segment.
7. **Risk & Priority Scoring** — An XGBoost regression/classification model consumes severity + contextual features (road age, traffic load, prior history) to output a maintenance priority score / estimated risk.
8. **Explainability** — SHAP computes per-prediction feature attributions so engineers can see *why* a segment was flagged high-priority.
9. **Maintenance Recommendation** — The decision engine combines risk score + business rules (budget, thresholds) into an actionable recommendation (e.g., "Immediate repair," "Schedule within 30 days," "Monitor").
10. **Alerting** — Any detection crossing the critical severity threshold raises an entry in `alerts` for immediate visibility on the dashboard.
11. **Persistence** — All intermediate and final artifacts (detections, features, scores, SHAP values, recommendations) are written to SQLite.
12. **Visualization** — Streamlit dashboard renders KPIs, geo-map of road/flyover health, severity trends, and priority rankings using Plotly.
13. **Reporting** — On demand, ReportLab compiles a formatted PDF (inspection summary, annotated images, severity breakdown, SHAP explanation snapshots, recommended actions) for stakeholders/authorities.
14. **Feedback Loop (optional/offline)** — Verified outcomes (actual repair costs, failure events) are fed back into `training/` scripts to periodically retrain severity/priority models.

---

## 5. Module Responsibilities

| Module | Responsibility |
|---|---|
| `app/` | All Streamlit UI: pages, navigation, widgets. Contains no business logic — only calls into `core/`. |
| `core/detection/` | Wraps YOLOv8 model loading and inference; handles both image and video (frame-by-frame) paths; produces raw detection records. |
| `core/severity/` | Converts raw detections into engineering-meaningful severity metrics and classifications. |
| `core/decision_intelligence/` | Houses the XGBoost priority model, SHAP explainability, composite risk scoring, and the rules engine that turns scores into recommendations. |
| `core/reporting/` | Generates all outward-facing artifacts: Plotly figures for the dashboard and ReportLab PDFs for export. |
| `core/utils/` | Shared, stateless helpers (geo calculations, image/video I/O helpers, input validation). |
| `database/` | Owns all SQLite access — schema definition, migrations, and a single `db_manager` façade so no other module talks to SQLite directly. |
| `models/` | Static model artifacts (YOLO weights, trained sklearn/XGBoost models, scalers). Read-only at runtime. |
| `data/` | All datasets and pipeline-generated media/exports, separated into raw/processed/training/exports. |
| `training/` | Offline scripts for training/retraining severity and priority models; not invoked during normal app runtime. |
| `tests/` | Unit/integration tests per core module. |
| `config.py` | Single source of truth for paths, thresholds, model parameters, and environment-driven settings. |

---

## 6. Database Schema (SQLite)

```sql
-- Roads / Flyovers being monitored
CREATE TABLE roads (
    road_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    asset_type TEXT CHECK(asset_type IN ('road','flyover')) NOT NULL,
    location TEXT,
    latitude REAL,
    longitude REAL,
    length_km REAL,
    construction_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One inspection event (a batch of uploaded media)
CREATE TABLE inspection_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    road_id INTEGER NOT NULL,
    inspector_name TEXT,
    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    media_type TEXT CHECK(media_type IN ('image','video')) NOT NULL,
    status TEXT CHECK(status IN ('pending','processing','completed','failed')) DEFAULT 'pending',
    FOREIGN KEY (road_id) REFERENCES roads(road_id)
);

-- Individual uploaded files
CREATE TABLE media_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    file_type TEXT CHECK(file_type IN ('image','video')) NOT NULL,
    file_path TEXT NOT NULL,
    gps_lat REAL,
    gps_lon REAL,
    captured_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES inspection_sessions(session_id)
);

-- Damage type taxonomy
CREATE TABLE damage_types (
    damage_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,          -- e.g. crack, pothole, spalling, joint_failure
    description TEXT,
    base_severity_weight REAL DEFAULT 1.0
);

-- Raw YOLOv8 detections
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    frame_number INTEGER,               -- NULL for images
    damage_type_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    damage_area_px REAL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES media_files(file_id),
    FOREIGN KEY (damage_type_id) REFERENCES damage_types(damage_type_id)
);

-- Severity assessment per detection/segment
CREATE TABLE severity_assessments (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    severity_score REAL NOT NULL,        -- continuous 0-1 or 0-100
    severity_class TEXT CHECK(severity_class IN ('low','medium','high','critical')),
    area_percentage REAL,
    assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id)
);

-- ML-driven priority/risk predictions (XGBoost)
CREATE TABLE ml_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,            -- e.g. 'xgboost_priority_v1'
    predicted_priority_score REAL,
    predicted_risk_class TEXT,
    features_json TEXT,                  -- serialized feature vector used
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES inspection_sessions(session_id)
);

-- SHAP explainability values tied to a prediction
CREATE TABLE shap_explanations (
    explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    shap_value REAL NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES ml_predictions(prediction_id)
);

-- Final actionable recommendations
CREATE TABLE maintenance_recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    road_id INTEGER NOT NULL,
    prediction_id INTEGER,
    priority_level TEXT CHECK(priority_level IN ('low','medium','high','urgent')),
    recommended_action TEXT,
    estimated_cost REAL,
    due_date DATE,
    status TEXT CHECK(status IN ('open','in_progress','resolved')) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (road_id) REFERENCES roads(road_id),
    FOREIGN KEY (prediction_id) REFERENCES ml_predictions(prediction_id)
);

-- Generated PDF/exports
CREATE TABLE reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    report_type TEXT CHECK(report_type IN ('inspection','decision_summary')) NOT NULL,
    file_path TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES inspection_sessions(session_id)
);

-- System alerts for critical findings
CREATE TABLE alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    road_id INTEGER NOT NULL,
    detection_id INTEGER,
    severity TEXT,
    message TEXT,
    is_resolved BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (road_id) REFERENCES roads(road_id),
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id)
);

-- App users (optional access control)
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT CHECK(role IN ('admin','inspector','viewer')) DEFAULT 'viewer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Entity relationships (summary):**
`roads (1) → (N) inspection_sessions → (N) media_files → (N) detections → (1) severity_assessments`
`inspection_sessions (1) → (N) ml_predictions → (N) shap_explanations`
`roads (1) → (N) maintenance_recommendations`, `roads (1) → (N) alerts`, `inspection_sessions (1) → (N) reports`

---

## 7. Key Design Decisions

- **Model caching**: YOLOv8 and ML models loaded once via `st.cache_resource` to avoid reload on every Streamlit rerun.
- **Separation of concerns**: `core/` has zero Streamlit imports — keeps business logic testable and reusable (e.g., if later exposed via an API).
- **Explainability as a first-class citizen**: SHAP outputs are persisted (not just computed transiently) so historical decisions remain auditable.
- **Hybrid decision logic**: Maintenance recommendations combine ML priority scores with configurable business rules/thresholds — not a pure black-box output — for accountability with civic authorities.
- **Extensibility**: Schema and folder structure allow adding new damage classes, new ML models, or an API layer without restructuring.
