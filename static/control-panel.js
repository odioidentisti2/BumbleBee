const page = (document.body.dataset.pageLabel);

// Initialize control panel
function initControlsPanel(panel) {

  panel.innerHTML = `
    <input id="control-input" class="control-file-input" type="file" accept=".csv">
    <div class="control-file-row">
      <label for="control-input" class="control-file-label control-buttons">
        <svg viewBox="0 0 24 24"><path d="M4 7h6l2 2h8v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7z"></path></svg>
      </label>
      <span class="control-file-caption">Browse...</span>
    </div>
    <button type="button" id="controlButton" class="control-run-button control-buttons" disabled>${page}</button>
  `;

  const fileInput = panel.querySelector(".control-file-input");

  // Send selected file to the preview modal flow
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (!file) return;

    // runButton.disabled = true;

    document.dispatchEvent(new CustomEvent("csv:file-selected", {
      detail: { file },
    }));
  });
  
}

//POSSO ELIMINARLA DATO CHE HO UN SOLO PANNELLO?
// Initialize all reusable controls panels on the current page.
function initControlsPanels() {
  document.querySelectorAll(".controls").forEach(initControlsPanel);
}

initControlsPanels();

const consolePrompt = document.querySelector(".console-prompt")
const fileLabel = document.querySelector(".control-file-label");
const runButton = document.querySelector(".control-run-button");
window.csvInfo = null;

// Listen for CSV info ready.
document.addEventListener("csv:info-ready", async (event) => {
  window.csvInfo = event.detail.csvInfo
  consolePrompt.innerHTML = `Dataset: ${window.csvInfo.file_name}`;  // Update the console prompt
  runButton.disabled = false;  // Enable the RUN button
});


// Run
document
    .getElementById("controlButton")
    .addEventListener("click", runAction);

async function runAction() {
    console.log("runAction called");  // DEBUG
    consolePrompt.innerHTML = "";
    fileLabel.classList.add("is-disabled");
    runButton.disabled = true

    const endpoint = page === "TRAIN"
        ? "/api/train"
        : "/api/test";

    const response = await fetch(endpoint, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(window.csvInfo)
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.detail || "Operation failed");
    }

    console.log(result);

    fileLabel.classList.remove("is-disabled");
    runButton.disabled = false
}
