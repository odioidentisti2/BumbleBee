// Set the visible console text and update default/custom state.
function setConsoleText(panel, html, isDefault) {
  const textNode = panel.querySelector(".console-text");
  if (!textNode) {
    return;
  }
  textNode.innerHTML = html;
  if (isDefault) {
    panel.classList.add("console-default");
  } else {
    panel.classList.remove("console-default");
  }
}

// Initialize one console panel from data attributes.
function initConsolePanel(panel) {
  const blinkTarget = panel.dataset.blinkTarget || "";

  const htmlText = blinkTarget ? `Upload a <span class="blink">${blinkTarget}</span>...` : "";

  panel.innerHTML = `
    <div class="console-text">${htmlText}</div>
  `;

  if (blinkTarget) {
    panel.classList.add("console-default");
  }

  // Listen for updates from linked controls panels.
  panel.addEventListener("console:set-text", (event) => {
    const detail = event.detail || {};
    setConsoleText(panel, detail.html || "", Boolean(detail.isDefault));
  });
}

// Initialize all reusable console panels on the current page.
function initConsolePanels() {
  document.querySelectorAll(".console").forEach(initConsolePanel);
}

initConsolePanels();
