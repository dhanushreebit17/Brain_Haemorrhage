// ---------------------------------------------------------------------------
// config.js
// Fill these in with your own Supabase project details and backend URL.
// Get SUPABASE_URL and SUPABASE_ANON_KEY from: Supabase Dashboard -> Project
// Settings -> API. The anon key is safe to expose in frontend code (it only
// works within the Row Level Security rules defined in supabase/schema.sql).
// ---------------------------------------------------------------------------
window.CONFIG = {
  SUPABASE_URL: "https://YOUR-PROJECT-REF.supabase.co",
  SUPABASE_ANON_KEY: "YOUR-SUPABASE-ANON-KEY",

  // Where your Flask backend (app.py) is running.
  // Local dev default. Change to your deployed backend URL in production.
  API_BASE_URL: "http://localhost:5000",

  // Storage bucket names (must match supabase/schema.sql)
  BUCKET_CT_SCANS: "ct-scans",
  BUCKET_HEATMAPS: "heatmaps",
};
