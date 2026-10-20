const runButton = document.getElementById("controlButton");
const nextButton = document.querySelector(".nav-next");
const prompt = document.querySelector(".console-prompt");
const regions = document.querySelectorAll(".controls, .nav-bar");
const setRunning = (running) => regions.forEach((el) => (el.inert = running));

let dataset = null;
let model = null;
const endpoint = `/api/${document.body.dataset.page}`;

function refreshUI() {
  runButton.disabled = true;
  prompt.textContent = "";

  const page = document.body.dataset.page;
  const lines = [];

  if (page === "train") {
    if (dataset) {
      lines.push(`Dataset: ${dataset.file_name}`);
      runButton.disabled = false;
    }
    prompt.textContent = lines.join("\n");
    return;
  }

  if (page !== "test") return;

  if (dataset) {
    lines.push(`Dataset: ${dataset.file_name}`);
  }

  if (model) {
    lines.push(`Model: ${model.file_name}`);
  }

  if (dataset && model) {
    runButton.disabled = false;
  } else if (!dataset && !model) {
    lines.push("\n> Upload a DATASET and a MODEL");
  } else if (!dataset) {
    lines.push("\n> Upload a DATASET");
  } else {
    lines.push("\n> Upload a MODEL");
  }

  prompt.textContent = lines.join("\n");
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