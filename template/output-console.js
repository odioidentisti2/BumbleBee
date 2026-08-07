// Initialize one console panel from data attributes.
function initConsolePanel(panel) {
  const browseLabel = panel.dataset.browseLabel || "Upload a";
  const blinkTarget = panel.dataset.blinkTarget || "";

  const defaultPrompt = blinkTarget ? `${browseLabel} <span class="blink">${blinkTarget}</span>` : "";

  panel.innerHTML = `
    <div class="console-text">${defaultPrompt}</div>
  `;
}

// Initialize all reusable console panels on the current page.
function initConsolePanels() {
  document.querySelectorAll(".console").forEach(initConsolePanel);
}

initConsolePanels();
