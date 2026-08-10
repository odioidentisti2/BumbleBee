// Initialize one console panel from data attributes.
function initConsolePanel(panel) {
  const prompt = panel.dataset.prompt || "> Upload a";
  const blinkTarget = panel.dataset.blinkTarget || "DATASET";

  panel.innerHTML = `
    <div class="console-text">${prompt} <span class="blink">${blinkTarget}</span></div>
  `;
}

// Initialize all reusable console panels on the current page.
function initConsolePanels() {
  document.querySelectorAll(".console").forEach(initConsolePanel);
}

initConsolePanels();
