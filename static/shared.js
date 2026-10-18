// Footer navigation buttons
document.querySelectorAll(".nav-bar .nav-button[data-href]").forEach((button) => {
  button.addEventListener("click", () => {
    window.location.href = button.dataset.href;
  });
});
