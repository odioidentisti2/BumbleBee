// Shared page chrome: top menu bar and bottom status bar.
const header = document.getElementById("shared-header");
const footer = document.getElementById("shared-footer");
const label = document.body.dataset.pageLabel;
const device = document.body.dataset.device;

if (header) {
  header.className = "menu-bar";
  header.innerHTML = `<a href="/">
                        <img src="static/assets/Tox-Bumblebee.svg" alt="BumbleBee logo" class="logo">
                      </a>
                      ${label ? `<span class="page-label">${label}</span>` : ""}`;
}

if (footer) {
  footer.className = "status-bar";
  footer.innerHTML = `<span>Device: ${device}</span>`;
}

// Shared split animation trigger for pages using the split-ready class.
requestAnimationFrame(() => {
  document.body.classList.add("split-ready");
});