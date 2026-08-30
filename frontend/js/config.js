// ---------------------------------------------------------------------------
// config.js
// Fill these in with your own Supabase project details and backend URL.
// Get SUPABASE_URL and SUPABASE_ANON_KEY from: Supabase Dashboard -> Project
// Settings -> API. The anon key is safe to expose in frontend code (it only
// works within the Row Level Security rules defined in supabase/schema.sql).
// ---------------------------------------------------------------------------
window.CONFIG = {
  SUPABASE_URL: "https://kwvxtcnzmfdgehidrtib.supabase.co",
  SUPABASE_ANON_KEY: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt3dnh0Y256bWZkZ2VoaWRydGliIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgwNjk3MjksImV4cCI6MjEwMzY0NTcyOX0.fFFaCGrGRtWCbv2Gn0xaGfH1Xv5fhS0wQglaV1fps2w",

  // Where your Flask backend (app.py) is running.
  // Local dev default. Change to your deployed backend URL in production.
  API_BASE_URL: "http://localhost:5000",

  // Storage bucket names (must match supabase/schema.sql)
  BUCKET_CT_SCANS: "ct-scans",
  BUCKET_HEATMAPS: "heatmaps",
};
