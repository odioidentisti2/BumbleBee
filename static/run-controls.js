const runButton = document.getElementById("controlButton");
const nextButton = document.querySelector(".nav-next");
const prompt = document.querySelector(".console-prompt");
const regions = document.querySelectorAll(".controls, .nav-bar");
const setRunning = (running) => regions.forEach((el) => (el.inert = running));

let dataset = null;
let model = null;
const endpoint = `/api/${document.body.dataset.page}`;

function appendLine(parts) {
  if (prompt.childNodes.length) {
    prompt.appendChild(document.createElement("br"));
  }

  parts.forEach((part) => {
    if (typeof part === "string") {
      prompt.appendChild(document.createTextNode(part));
      return;
    }

    const span = document.createElement("span");
    span.className = "blink";
    span.textContent = part.blink;
    prompt.appendChild(span);
  });
}

function refreshUI() {
  runButton.disabled = true;
  prompt.textContent = "";

  const page = document.body.dataset.page;

  if (page === "train") {
    if (dataset) {
      appendLine([`Dataset: ${dataset.file_name}`]);
      runButton.disabled = false;
    } else {
      appendLine(["> Upload a ", { blink: "DATASET" }]);
    }
    return;
  }

  if (page !== "test") return;

  if (dataset) {
    appendLine([`Dataset: ${dataset.file_name}`]);
  }

  if (model) {
    appendLine([`Model: ${model.file_name}`]);
  }

  if (dataset && model) {
    runButton.disabled = false;
    return;
  } else if (!dataset && !model) {
    appendLine(["> Upload a ", { blink: "DATASET and a MODEL" }]);
  } else if (!dataset) {
    appendLine(["> Upload a ", { blink: "DATASET" }]);
  } else {
    appendLine(["> Upload a ", { blink: "MODEL" }]);
  }
}

document.addEventListener("dataset:selected", (event) => {
  dataset = event.detail;
  refreshUI();
});

document.addEventListener("model:selected", (event) => {
  model = event.detail;
  refreshUI();
});

runButton.addEventListener("click", async () => {
  if (!dataset) return;

  setRunning(true);
  prompt.textContent = "";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...dataset, model_file_name: model?.file_name }),
    });

    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      throw new Error(result.detail || "Operation failed");
    }

    nextButton.disabled = false;
    document.dispatchEvent(new CustomEvent("run:succeeded"));
  } catch (error) {
    console.error(error);
    document.dispatchEvent(new CustomEvent("run:failed", { detail: { error } }));
  } finally {
    setRunning(false);
  }
});