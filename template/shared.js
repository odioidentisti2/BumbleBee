const header = document.getElementById("shared-header");
const footer = document.getElementById("shared-footer");
const label = document.body.dataset.pageLabel;
const device = document.body.dataset.device;

if (header) {
  header.className = "menu-bar";
  header.innerHTML = `<img src="assets/BB_logo_iceland.png" alt="BumbleBee logo" class="logo">${label ? `<span class="page-label">${label}</span>` : ""}`;
}

if (footer) {
  footer.className = "status-bar";
  footer.innerHTML = `<span>Device: ${device}</span>`;
}
