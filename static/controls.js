const runButton = document.getElementById("controlButton");
const nextButton = document.querySelector(".nav-next");
const regions = document.querySelectorAll(".controls, .nav-bar");
const setRunning = (running) => regions.forEach((el) => (el.inert = running));

let dataset = null;
let model = null;
const endpoint = `/api/${document.body.dataset.page}`;


function refreshControls() {
  const page = document.body.dataset.page;
  const ready = page === "train" ? Boolean(dataset) : Boolean(dataset && model);
  runButton.disabled = !ready;
  // nextButton.disabled = true;
}

// Ceheck if model from training page to test page
if (document.body.dataset.hasTrainedModel === "true") {
  model = { trained: true };
  refreshControls();
  refreshConsole({ dataset, model });
}

document.addEventListener("dataset:selected", (event) => {
  dataset = event.detail;
  refreshControls();
  refreshConsole({ dataset, model });
});

document.addEventListener("model:selected", (event) => {
  model = event.detail;
  refreshControls();
  refreshConsole({ dataset, model });
});

runButton.addEventListener("click", async () => {
  if (!dataset) return;

  setRunning(true);
  clearConsole();

  const body = {
    dataset_info: dataset,
    // model_name: model?.file_name ?? null,
    model_name: model?.trained ? "FROM TRAIN PAGE" : (model?.file_name ?? null),
  };

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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