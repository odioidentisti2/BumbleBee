const page = (document.body.dataset.pageLabel);

// Initialize control panel
function initControlsPanel(panel) {

  panel.innerHTML = `
    <input id="control-input" class="control-file-input" type="file" accept=".csv">
    <!-- input linked to label -->
    <div class="control-file-row">
      <label for="control-input" class="control-file-button control-buttons">
        <svg viewBox="0 0 24 24"><path d="M4 7h6l2 2h8v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7z"></path></svg>
      </label>
      <span class="control-file-caption">Browse...</span>
    </div>
    <button type="button" id="controlButton" class="control-run-button control-buttons" disabled>${page}</button>
  `;

  const fileInput = panel.querySelector(".control-file-input");
  const fileButton = panel.querySelector(".control-file-button");
  const runButton = panel.querySelector(".control-run-button");

  // Send selected file to the preview modal flow
  fileInput.addEventListener("change", () => {
    const selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;

    runButton.disabled = true;

    document.dispatchEvent(new CustomEvent("csv:file-selected", {
      detail: {
        file: selectedFile,
      },
    }));
  });

  
}

// Initialize all reusable controls panels on the current page.
function initControlsPanels() {
  document.querySelectorAll(".controls").forEach(initControlsPanel);
}

initControlsPanels();


window.csvInfo = null;

// Listen for dataset-info ready.
document.addEventListener("csv:dataset-info-ready", async (event) => {
  window.csvInfo = event.detail.csvInfo
  document.querySelector(".console-prompt").innerHTML = `Dataset: ${window.csvInfo.path}`;  // Update the console prompt
  document.querySelector(".control-run-button").disabled = false;  // Enable the RUN button
});


// Run TRAIN
document
    .getElementById("controlButton")
    .addEventListener("click", runAction);

async function runAction() {
    console.log("runAction called");  // DEBUG
    document.querySelector(".console-prompt").innerHTML = "";

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
}
