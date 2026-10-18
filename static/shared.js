// Footer navigation buttons
document.querySelectorAll(".nav-bar .nav-button[data-href]").forEach((button) => {
  button.addEventListener("click", () => {
    window.location.href = button.dataset.href;
  });
});

// Split animation trigger for pages using the split-ready class.
requestAnimationFrame(() => {
  document.body.classList.add("split-ready");
});