// Build the folder icon used by the file picker trigger.
function buildFolderIcon() {
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h6l2 2h8v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7z"></path></svg>';
}

// Create the default prompt shown in the paired console before file selection.
function getDefaultPrompt(browseLabel, targetLabel) {
  return `${browseLabel} <span class="blink">${targetLabel}</span>...`;
}

// Initialize one controls panel from data attributes.
function initControlsPanel(panel) {
  const browseLabel = panel.dataset.browseLabel || "Upload a";
  const targetLabel = panel.dataset.targetLabel || "DATASET";
  const outputTarget = panel.dataset.outputTarget || "";

  panel.innerHTML = `
    <input class="control-file-input" type="file" accept=".csv">
    <label class="control-file-button">${buildFolderIcon()}</label>
  `;

  const fileInput = panel.querySelector(".control-file-input");
  const fileButton = panel.querySelector(".control-file-button");
  if (!fileInput || !fileButton) {
    return;
  }

  // Link label and input so the label opens the system file picker.
  const inputId = panel.id ? `${panel.id}-input` : `control-input-${Math.random().toString(36).slice(2)}`;
  fileInput.id = inputId;
  fileButton.setAttribute("for", inputId);

  // Push selected filename (or default prompt) to the target console panel.
  fileInput.addEventListener("change", () => {
    const selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0].name : "";
    const text = selectedFile || getDefaultPrompt(browseLabel, targetLabel);

    if (outputTarget) {
      const outputPanel = document.querySelector(outputTarget);
      if (outputPanel) {
        outputPanel.dispatchEvent(new CustomEvent("console:set-text", {
          detail: {
            html: text,
            isDefault: !selectedFile,
          },
        }));
      }
    }
  });
}

// Initialize all reusable controls panels on the current page.
function initControlsPanels() {
  document.querySelectorAll(".controls").forEach(initControlsPanel);
}

initControlsPanels();
