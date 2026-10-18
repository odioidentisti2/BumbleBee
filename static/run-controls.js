const runButton = document.getElementById("controlButton");
const nextButton = document.querySelector(".nav-next");
const prompt = document.querySelector(".console-prompt");
const regions = document.querySelectorAll(".controls, .nav-bar");
const setRunning = (running) => regions.forEach((el) => (el.inert = running));

let dataset = null;
const endpoint = `/api/${document.body.dataset.page}`;

document.addEventListener("dataset:selected", (event) => {
  dataset = event.detail;
  nextButton.disabled = true;
  prompt.textContent = `Dataset: ${dataset.file_name}`;
  runButton.disabled = false;
});

runButton.addEventListener("click", async () => {
  if (!dataset) return;

  setRunning(true);
  prompt.textContent = "";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dataset),
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