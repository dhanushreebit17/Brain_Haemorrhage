// ---------------------------------------------------------------------------
// supabaseClient.js
// Single shared Supabase client + small data-access helpers used by every
// page. Loaded after the Supabase UMD script and config.js.
// ---------------------------------------------------------------------------
const sb = window.supabase.createClient(
  window.CONFIG.SUPABASE_URL,
  window.CONFIG.SUPABASE_ANON_KEY
);

// ---- Auth helpers ---------------------------------------------------------

async function requireSession() {
  const { data: { session } } = await sb.auth.getSession();
  if (!session) {
    window.location.href = "index.html";
    return null;
  }
  const welcomeEl = document.querySelector("[data-welcome-name]");
  if (welcomeEl) {
    const label = session.user.user_metadata?.full_name || session.user.email || "Radiologist";
    welcomeEl.textContent = `Welcome, ${label}`;
  }
  return session;
}

async function logout() {
  await sb.auth.signOut();
  window.location.href = "index.html";
}

// ---- Scans table helpers ---------------------------------------------------

async function fetchDashboardStats() {
  const { count: total } = await sb.from("scans").select("*", { count: "exact", head: true });
  const { count: hemorrhage } = await sb.from("scans").select("*", { count: "exact", head: true }).eq("hemorrhage_detected", true);
  const { count: normal } = await sb.from("scans").select("*", { count: "exact", head: true }).eq("hemorrhage_detected", false);
  const { count: reports } = await sb.from("scans").select("*", { count: "exact", head: true }).eq("report_generated", true);
  return {
    total: total || 0,
    hemorrhage: hemorrhage || 0,
    normal: normal || 0,
    reports: reports || 0,
  };
}

async function fetchRecentScans(limit = 4) {
  const { data, error } = await sb
    .from("scans")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(limit);
  if (error) throw error;
  return data;
}

async function fetchScanHistory({ patientId = "", page = 1, pageSize = 5 } = {}) {
  let query = sb.from("scans").select("*", { count: "exact" }).order("created_at", { ascending: false });
  if (patientId.trim()) {
    query = query.ilike("patient_id", `%${patientId.trim()}%`);
  }
  const from = (page - 1) * pageSize;
  const to = from + pageSize - 1;
  const { data, error, count } = await query.range(from, to);
  if (error) throw error;
  return { rows: data, total: count || 0 };
}

async function fetchScanById(id) {
  const { data, error } = await sb.from("scans").select("*").eq("id", id).single();
  if (error) throw error;
  return data;
}

async function uploadImageToStorage(bucket, path, base64Png) {
  const bytes = Uint8Array.from(atob(base64Png), (c) => c.charCodeAt(0));
  const { error } = await sb.storage.from(bucket).upload(path, bytes, {
    contentType: "image/png",
    upsert: true,
  });
  if (error) throw error;
  const { data } = sb.storage.from(bucket).getPublicUrl(path);
  return data.publicUrl;
}

async function insertScanRecord(record) {
  const { data: { session } } = await sb.auth.getSession();
  const { data, error } = await sb
    .from("scans")
    .insert({ ...record, created_by: session?.user?.id })
    .select()
    .single();
  if (error) throw error;
  return data;
}

async function markReportGenerated(id) {
  await sb.from("scans").update({ report_generated: true }).eq("id", id);
}
