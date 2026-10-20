
const input = document.getElementById("control-input");
const label = document.querySelector(".control-file-label");

const DESCRIPTIONS = {
  ".csv": { description: "Dataset files", accept: { "text/csv": [".csv"] } },
  ".pt": { description: "Model files", accept: { "application/octet-stream": [".pt"] } },
};

const types = input.accept
  .split(",")
  .map((ext) => DESCRIPTIONS[ext])
  .filter(Boolean);

async function pick() {
  if (!window.showOpenFilePicker || !types.length) {
    input.click();
    return;
  }

  try {
    const [handle] = await window.showOpenFilePicker({ types });
    const data = new DataTransfer();
    data.items.add(await handle.getFile());
    input.files = data.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  } catch (error) {
    if (error.name !== "AbortError") throw error;
  }
}

label.addEventListener("click", (event) => {
  event.preventDefault();
  pick();
});

input.addEventListener("change", () => {
  const file = input.files?.[0];
  if (!file || !file.name.toLowerCase().endsWith(".pt")) return;

  input.value = "";
  document.dispatchEvent(new CustomEvent("model:selected", {
    detail: { file_name: file.name },
  }));
});