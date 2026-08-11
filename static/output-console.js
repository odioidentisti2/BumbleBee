let consoleEventSource = null;

// Initialize one console panel from data attributes
function initConsolePanel(panel) {
  const prompt = panel.dataset.prompt || "> Upload a";
  const blinkTarget = panel.dataset.blinkTarget || "DATASET";

  panel.innerHTML = `
    <div class="console-prompt">${prompt} <span class="blink">${blinkTarget}</span></div>
    <pre class="console-log"></pre>
  `;
}

// Initialize all reusable console panels on the current page
function initConsolePanels() {
  document.querySelectorAll(".console").forEach(initConsolePanel);
  startConsoleStream();
}

initConsolePanels();


// Console log
function appendConsoleLine(line) {
  const consolePanel = document.querySelector(".console");
  if (!consolePanel) {
    return;
  }

  let logArea = consolePanel.querySelector(".console-log");
  if (!logArea) {
    logArea = document.createElement("pre");
    logArea.className = "console-log";
    consolePanel.appendChild(logArea);
  }

  logArea.textContent += `${line}\n`;
  logArea.scrollTop = logArea.scrollHeight;
}

function startConsoleStream() {
  if (consoleEventSource) {
    return;
  }

  consoleEventSource = new EventSource("/api/logs");
  consoleEventSource.onmessage = (event) => {
    appendConsoleLine(event.data);
  };
}
