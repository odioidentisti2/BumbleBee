// Set the visible console text.
function setConsoleText(panel, html) {
  const textNode = panel.querySelector(".console-text");
  if (!textNode) {
    return;
  }
  textNode.innerHTML = html;
}

// Initialize one console panel from data attributes.
function initConsolePanel(panel) {
  const browseLabel = panel.dataset.browseLabel || "Upload a";
  const blinkTarget = panel.dataset.blinkTarget || "";

  const defaultPrompt = blinkTarget ? `${browseLabel} <span class="blink">${blinkTarget}</span>` : "";

  panel.innerHTML = `
    <div class="console-text">${defaultPrompt}</div>
  `;

  // Listen for updates from linked controls panels.
  panel.addEventListener("console:set-text", (event) => {
    const detail = event.detail || {};
    setConsoleText(panel, detail.selected ? "" : defaultPrompt);
  });
}

// Initialize all reusable console panels on the current page.
function initConsolePanels() {
  document.querySelectorAll(".console").forEach(initConsolePanel);
}

initConsolePanels();
