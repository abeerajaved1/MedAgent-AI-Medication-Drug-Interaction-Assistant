/* Shared "logged-in app" shell: left sidebar + topbar, injected into every
   protected page's #sidebar-slot / #topbar-slot. Kept as plain DOM string
   templating (no framework) so every HTML page stays a simple static file
   the FastAPI app can serve directly. */

const NAV_LINKS = [
  { href: "dashboard.html", icon: "home", label: "Dashboard" },
  { href: "assistant.html", icon: "chat", label: "AI Medical Assistant" },
  { href: "medicine.html", icon: "pill", label: "Ask About Medicine" },
  { href: "analyze-image.html", icon: "image", label: "Analyze Image" },
  { href: "interactions.html", icon: "interactions", label: "Drug Interactions" },
];
const NAV_LINKS_2 = [
  { href: "history.html", icon: "history", label: "History" },
  { href: "history.html?saved=1", icon: "bookmark", label: "Saved Results" },
];

function currentPage() {
  return location.pathname.split("/").pop() || "dashboard.html";
}

function toast(message, type = "") {
  let stack = document.querySelector(".toast-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.className = "toast-stack";
    document.body.appendChild(stack);
  }
  const el = document.createElement("div");
  el.className = "toast" + (type ? " " + type : "");
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}
window.toast = toast;

function initials(name) {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/);
  return (parts[0][0] + (parts[1] ? parts[1][0] : "")).toUpperCase();
}

function requireAuth() {
  // Guests are allowed to browse — session_id still works for
  // anonymous use (see MedAPI.getSessionId) — but personalized chrome
  // (name, avatar) only renders for logged-in users. If you want to
  // hard-require login instead, uncomment the redirect below.
  const user = MedAPI.getUser();
  return user;
}

function renderSidebar(active) {
  const slot = document.getElementById("sidebar-slot");
  if (!slot) return;
  const linkHtml = (l) => {
    const isActive = active === l.href.split("?")[0];
    return `<a class="nav-link ${isActive ? "active" : ""}" href="${l.href}">${Icons[l.icon]}<span>${l.label}</span></a>`;
  };
  slot.innerHTML = `
    <div class="brand" style="padding:6px 8px 22px;">
      <div class="brand-mark">${Icons.shield}</div>
      <div class="brand-text">
        <div class="name">Med<span>Agent</span></div>
        <div class="tagline">AI Pharmacist &amp; Diagnostic Assistant</div>
      </div>
    </div>
    <nav class="nav-section">${NAV_LINKS.map(linkHtml).join("")}</nav>
    <div class="nav-divider"></div>
    <nav class="nav-section">${NAV_LINKS_2.map(linkHtml).join("")}</nav>
    <div class="nav-divider"></div>
    <nav class="nav-section">
      <a class="nav-link" href="profile.html">${Icons.user}<span>Profile &amp; Settings</span></a>
      <a class="nav-link" href="help.html">${Icons.help}<span>Help &amp; Support</span></a>
    </nav>
    <div class="grow"></div>
    <nav class="nav-section">
      <a class="nav-link nav-logout" href="#" id="sidebar-logout">${Icons.logout}<span>Logout</span></a>
    </nav>
  `;
  document.getElementById("sidebar-logout").addEventListener("click", async (e) => {
    e.preventDefault();
    await MedAPI.logout();
    location.href = "login.html";
  });
}

function renderTopbar({ showSearch = true } = {}) {
  const slot = document.getElementById("topbar-slot");
  if (!slot) return;
  const user = MedAPI.getUser();
  const name = user ? user.name : "Guest";
  const searchHtml = showSearch ? `
    <div class="input-wrap grow" style="max-width:520px;">
      <span class="leading">${Icons.search}</span>
      <input class="input" id="global-search" placeholder="Search medicines, symptoms, or ask anything..." />
    </div>` : `<div class="grow"></div>`;
  slot.innerHTML = `
    ${searchHtml}
    <button class="btn btn-ghost" id="notif-btn" style="position:relative;padding:8px;">
      ${Icons.bell}
      <span class="badge badge-red" style="position:absolute;top:-2px;right:-2px;padding:1px 5px;">3</span>
    </button>
    <div class="flex items-center gap-10" style="cursor:pointer;position:relative;" id="user-menu-trigger">
      <div class="avatar">${initials(name)}</div>
      <div>
        <div class="small bold" style="line-height:1.2;">${name}</div>
        <div class="tiny muted" style="line-height:1.2;">Patient</div>
      </div>
      ${Icons.chevronDown}
    </div>
  `;
  const search = document.getElementById("global-search");
  if (search) {
    search.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && search.value.trim()) {
        location.href = "assistant.html?q=" + encodeURIComponent(search.value.trim());
      }
    });
  }
  const menuTrigger = document.getElementById("user-menu-trigger");
  if (menuTrigger) {
    menuTrigger.addEventListener("click", () => {
      if (confirm("Log out of MedAgent?")) {
        MedAPI.logout().then(() => (location.href = "login.html"));
      }
    });
  }
}

function initShell(opts = {}) {
  renderSidebar(currentPage());
  renderTopbar(opts);
}
window.initShell = initShell;
window.requireAuth = requireAuth;
