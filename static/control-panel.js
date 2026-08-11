window.bumblebeeDatasetConfig = null;

// Initialize one controls panel from data attributes.
function initControlsPanel(panel) {
  const pageLabel = (document.body.dataset.pageLabel || "RUN");

  panel.innerHTML = `
    <input class="control-file-input" type="file" accept=".csv">
    <div class="control-file-row">
      <label class="control-file-button control-buttons">
        <svg viewBox="0 0 24 24"><path d="M4 7h6l2 2h8v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7z"></path></svg>
      </label>
      <span class="control-file-caption">Browse...</span>
    </div>
    <button type="button" id="controlButton" class="control-action-button control-buttons" disabled>${pageLabel}</button>
  `;

  const fileInput = panel.querySelector(".control-file-input");
  const fileButton = panel.querySelector(".control-file-button");
  const actionButton = panel.querySelector(".control-action-button");
  if (!fileInput || !fileButton || !actionButton) {
    return;
  }

  // Link label and input so the label opens the system file picker.
  const inputId = panel.id ? `${panel.id}-input` : `control-input-${Math.random().toString(36).slice(2)}`;
  fileInput.id = inputId;
  fileButton.setAttribute("for", inputId);

  // Send selected file to the preview modal flow.
  fileInput.addEventListener("change", () => {
    const selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
    if (!selectedFile) {
      return;
    }

    actionButton.disabled = true;

    document.dispatchEvent(new CustomEvent("csv:file-selected", {
      detail: {
        file: selectedFile,
      },
    }));
  });

  // Clear the console text and enable the action button when the config is ready.
  document.addEventListener("csv:config-ready", (event) => {
    console.log("config-ready received");  // DEBUG
    actionButton.disabled = false;
    window.bumblebeeDatasetConfig = event.detail?.config || null;

   const consolePrompt = document.querySelector(".console-prompt");
   if (consolePrompt) {
     consolePrompt.innerHTML = "";
   }
  });
}

// Initialize all reusable controls panels on the current page.
function initControlsPanels() {
  document.querySelectorAll(".controls").forEach(initControlsPanel);
}

initControlsPanels();

// Run
document
    .getElementById("controlButton")
    .addEventListener("click", runAction);

async function runAction() {
    console.log("runAction called");  // DEBUG
    const page = document.body.dataset.pageLabel; // "train" or "test"
    const config = window.bumblebeeDatasetConfig;

    if (!config) {
        throw new Error("Dataset configuration is not ready");  // DEBUG
    }

    const endpoint = page === "TRAIN"
        ? "/api/train"
        : "/api/test";

    const response = await fetch(endpoint, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(config)
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.detail || "Operation failed");
    }

    console.log(result);
}
