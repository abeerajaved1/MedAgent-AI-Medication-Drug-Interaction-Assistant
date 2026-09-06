/* =========================================================================
   MedAgent — API client
   Thin fetch wrapper around the FastAPI backend. No build step / bundler:
   loaded as a plain <script> on every page, exposes `window.MedAPI`.
   ========================================================================= */

(function () {
  const BASE = ""; // same-origin — the FastAPI app serves these static files itself

  function getToken() {
    return localStorage.getItem("medagent_token") || "";
  }
  function setToken(token) {
    if (token) localStorage.setItem("medagent_token", token);
  }
  function clearToken() {
    localStorage.removeItem("medagent_token");
    localStorage.removeItem("medagent_user");
  }
  function getUser() {
    try { return JSON.parse(localStorage.getItem("medagent_user") || "null"); }
    catch { return null; }
  }
  function setUser(user) {
    localStorage.setItem("medagent_user", JSON.stringify(user));
  }
  // The auth token doubles as session_id everywhere (chat memory, profile,
  // history, findings) — see app/auth.py. Guests who never log in still
  // get a stable per-browser session id so features still work.
  function getSessionId() {
    let sid = getToken();
    if (sid) return sid;
    sid = localStorage.getItem("medagent_guest_sid");
    if (!sid) {
      sid = "guest_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem("medagent_guest_sid", sid);
    }
    return sid;
  }

  async function request(path, { method = "GET", body, auth = true, isForm = false } = {}) {
    const headers = {};
    if (!isForm) headers["Content-Type"] = "application/json";
    if (auth && getToken()) headers["Authorization"] = "Bearer " + getToken();

    const res = await fetch(BASE + path, {
      method,
      headers,
      body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
    });

    let data = null;
    try { data = await res.json(); } catch { /* no body */ }

    if (!res.ok) {
      const message = (data && (data.detail || data.message)) || `Request failed (${res.status})`;
      const err = new Error(typeof message === "string" ? message : JSON.stringify(message));
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  const MedAPI = {
    getToken, setToken, clearToken, getUser, setUser, getSessionId,
    isLoggedIn: () => !!getToken(),

    // --- auth ---
    register: (name, email, password) => request("/api/auth/register", { method: "POST", body: { name, email, password }, auth: false }),
    login: (email, password) => request("/api/auth/login", { method: "POST", body: { email, password }, auth: false }),
    me: () => request("/api/auth/me"),
    logout: async () => { try { await request("/api/auth/logout", { method: "POST" }); } catch {} clearToken(); },

    // --- chat / assistant ---
    chat: (message) => request("/api/chat", { method: "POST", body: { message, session_id: getSessionId() } }),
    getTrace: (queryId) => request(`/api/trace/${encodeURIComponent(queryId)}`),

    // --- medicines ---
    searchMedicines: (q) => request(`/api/medicines${q ? "?q=" + encodeURIComponent(q) : ""}`, { auth: false }),
    getMedicineDetail: (name) => request(`/api/medicines/${encodeURIComponent(name)}`, { auth: false }),

    // --- interactions ---
    analyzeInteraction: (drugA, drugB) => request("/api/interactions/analyze", {
      method: "POST", body: { drug_a: drugA, drug_b: drugB, session_id: getSessionId() },
    }),

    // --- symptom checker ---
    symptomCheck: (description) => request("/api/symptom-check", {
      method: "POST", body: { description, session_id: getSessionId() },
    }),

    // --- vision / OCR ---
    analyzeImage: (file) => {
      const form = new FormData();
      form.append("file", file);
      form.append("session_id", getSessionId());
      return request("/api/vision", { method: "POST", body: form, isForm: true });
    },
    readPrescription: (file) => {
      const form = new FormData();
      form.append("file", file);
      form.append("session_id", getSessionId());
      return request("/api/ocr", { method: "POST", body: form, isForm: true });
    },

    // --- profile ---
    getProfile: () => request(`/api/profile/${encodeURIComponent(getSessionId())}`).catch((e) => {
      if (e.status === 404) return null;
      throw e;
    }),
    saveProfile: (fields) => request("/api/profile", { method: "POST", body: { session_id: getSessionId(), ...fields } }),

    // --- history / dashboard ---
    getDashboard: () => request(`/api/dashboard/${encodeURIComponent(getSessionId())}`),
    getHistory: (type, savedOnly) => {
      const params = new URLSearchParams();
      if (type && type !== "all") params.set("type", type);
      if (savedOnly) params.set("saved_only", "true");
      return request(`/api/history/${encodeURIComponent(getSessionId())}?${params.toString()}`);
    },
    toggleSaveHistory: (id) => request(`/api/history/${id}/toggle-save?session_id=${encodeURIComponent(getSessionId())}`, { method: "POST" }),
    deleteHistoryItem: (id) => request(`/api/history/${id}?session_id=${encodeURIComponent(getSessionId())}`, { method: "DELETE" }),

    // --- feedback ---
    sendFeedback: (query, rating, opts = {}) => request("/api/feedback", {
      method: "POST", body: { query, rating, session_id: getSessionId(), ...opts },
    }),
  };

  window.MedAPI = MedAPI;
})();
