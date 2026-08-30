# Brain Hemorrhage Detection System

A final-year project: a web app that takes a brain CT scan, runs it through a
DenseNet-121 classifier with Grad-CAM explainability, and lets a radiologist
log in, upload scans, review AI predictions with a heatmap overlay, and keep
a searchable history/report trail.

**Stack**
- **ML backend**: Flask + PyTorch (DenseNet-121) + Grad-CAM + OpenCV (`/backend`)
- **Frontend**: plain HTML/CSS/JS (`/frontend`) — no build step, open directly in a browser
- **Database + Auth + Storage**: Supabase (`/supabase/schema.sql`)

---

## 1. Project structure

```
brain-hemorrhage-detection/
├── backend/              # Flask ML API
│   ├── app.py             # /api/predict endpoint (upload → prediction + Grad-CAM)
│   ├── model.py            # DenseNet-121 model + class names
│   ├── gradcam.py          # Grad-CAM implementation
│   ├── train.py             # Fine-tunes on your dataset
│   ├── evaluate.py          # Confusion matrix / classification report
│   ├── prepare_dataset_v2.py# Dedup + train/val split from raw Kaggle data
│   ├── dicom_to_png.py      # DICOM → PNG conversion (windowing)
│   ├── requirements.txt
│   └── models/              # put haemorrhage_model.pth here after training
├── frontend/              # Static site — Login, Dashboard, Upload, Results, Report, History
│   ├── index.html            (Login)
│   ├── dashboard.html
│   ├── upload.html
│   ├── results.html
│   ├── report.html
│   ├── history.html
│   ├── css/style.css
│   └── js/
│       ├── config.js          # <-- fill in your Supabase URL/key here
│       ├── supabaseClient.js  # Supabase client + all data-access helpers
│       └── (auth handled inline in each page via supabaseClient.js)
├── supabase/
│   └── schema.sql          # scans table + RLS policies + storage buckets
└── .gitignore
```

---

## 2. Set up Supabase (5 minutes)

1. Create a free project at [supabase.com](https://supabase.com).
2. Go to **SQL Editor** → paste in the contents of `supabase/schema.sql` → **Run**.
   This creates the `scans` table, Row Level Security policies, and two
   storage buckets (`ct-scans`, `heatmaps`).
3. Go to **Authentication → Users → Add user** and create a login for
   yourself (e.g. `radiologist@example.com` / a password) — this is the
   account you'll log in with on the login page ("For Radiologists Only").
4. Go to **Project Settings → API** and copy your **Project URL** and
   **anon public key**.
5. Paste them into `frontend/js/config.js`:
   ```js
   window.CONFIG = {
     SUPABASE_URL: "https://xxxxxxxx.supabase.co",
     SUPABASE_ANON_KEY: "eyJhbGciOi...",
     API_BASE_URL: "http://localhost:5000",
     ...
   };
   ```

---

## 3. Run the backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

This starts the Flask API on `http://localhost:5000`.

- If `backend/models/haemorrhage_model.pth` doesn't exist yet, the app still
  runs end-to-end using an untrained classifier head (predictions won't be
  meaningful, but the full upload → predict → Grad-CAM → save pipeline
  works, which is enough to demo the app).
- To get real predictions: prepare your dataset with `prepare_dataset_v2.py`,
  then train with `train.py`, which writes the weights to
  `backend/models/haemorrhage_model.pth`.

**Important limitation to know for your report/viva:** `model.py`'s
`CLASS_NAMES` currently has 5 hemorrhage *subtypes* only (epidural,
intracerebral, intraventricular, subarachnoid, subdural) — there's no
`normal` class. That means the current model always predicts "Hemorrhage
Detected: Yes" plus a subtype; it can't yet tell a normal scan from an
abnormal one. To match the wireframe's "Normal Scans" stat properly, add a
`normal` class folder to your dataset and retrain. The frontend already
reads `hemorrhage_detected` as a stored boolean, so once the model can
predict "normal" you just need to set that field accordingly when saving
the scan (in `upload.html`'s upload handler).

Your `confusion_matrix.png` (included in `backend/`) shows ~54.8% validation
accuracy across the 5 subtypes — worth mentioning honestly in your report
as a starting point, with ideas for improvement (more data, class balancing,
a normal class, longer training, or a stronger backbone).

---

## 4. Run the frontend

No build step needed — but serve it over HTTP (not `file://`) so the
Supabase JS client and fetch calls behave correctly:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500` in your browser. Log in with the
Supabase user you created in step 2.

---

## 5. How the pieces connect

1. **Login** (`index.html`) → Supabase Auth (`signInWithPassword`).
2. **Upload Scan** (`upload.html`) → sends the image to the Flask backend
   (`POST /api/predict`) → backend returns the prediction, confidence, and
   both images (original + Grad-CAM overlay) as base64 → frontend uploads
   both images to Supabase Storage and inserts a row into the `scans` table.
3. **Dashboard** / **History** / **Results** / **Report** pages all read
   from the `scans` table directly via the Supabase JS client — no backend
   round-trip needed for viewing past scans.
4. **Report → Export as PDF** uses the browser's native print-to-PDF
   (`window.print()` with print-specific CSS in `style.css`) — no extra
   dependency needed.

---

## 6. Git

This repo is already initialized with git. To push it to GitHub:

```bash
git remote add origin https://github.com/<your-username>/brain-hemorrhage-detection.git
git branch -M main
git push -u origin main
```

Model weights (`.pth`), the local `dataset/` folder, and uploaded temp files
are excluded via `.gitignore` — don't commit large binaries to git; if you
need to share trained weights, use Git LFS or a cloud link instead.
