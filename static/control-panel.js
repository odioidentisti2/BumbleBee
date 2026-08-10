// Build the folder icon used by the file picker trigger.
function buildFolderIcon() {
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h6l2 2h8v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7z"></path></svg>';
}

// Initialize one controls panel from data attributes.
function initControlsPanel(panel) {
  const pageLabel = (document.body.dataset.pageLabel || "RUN").trim().toUpperCase();

  panel.innerHTML = `
    <input class="control-file-input" type="file" accept=".csv">
    <div class="control-file-row">
      <label class="control-file-button control-button">${buildFolderIcon()}</label>
      <span class="control-file-caption">Browse...</span>
    </div>
    <button type="button" class="control-action-button control-button" disabled>${pageLabel}</button>
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

  document.addEventListener("csv:config-ready", () => {
    actionButton.disabled = false;

    const consoleText = document.querySelector(".console-text");
    if (consoleText) {
      consoleText.innerHTML = "";
    }
  });
}

// Initialize all reusable controls panels on the current page.
function initControlsPanels() {
  document.querySelectorAll(".controls").forEach(initControlsPanel);
}

initControlsPanels();
