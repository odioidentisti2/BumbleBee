const trainingSetInput = document.getElementById("trainingSetInput");
const trainingSetText = document.getElementById("trainingSetText");

requestAnimationFrame(() => {
  document.body.classList.add("split-ready");
});

if (trainingSetInput && trainingSetText) {
  trainingSetInput.addEventListener("change", () => {
    const selectedFile = trainingSetInput.files && trainingSetInput.files[0] ? trainingSetInput.files[0].name : "";
    trainingSetText.textContent = selectedFile || "Browse a TRAINING SET...";
  });
}