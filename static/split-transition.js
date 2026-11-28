// Layout transition module: owns the transition CSS and the reload behaviour.
const css = `
  /* from-* only chooses the start ratio */
  .train-page.from-test  .main-split { --split-from: 1fr 2fr; }
  .test-page.from-train  .main-split { --split-from: 2fr 1fr; }
  .train-page:not(.from-test) .main-split,
  .test-page:not(.from-train) .main-split { --split-from: 1fr 1fr; }

  /* final layout */
  .train-page .main-split { --split-to: 2fr 1fr; grid-template-columns: var(--split-to); }
  .test-page  .main-split { --split-to: 1fr 2fr; grid-template-columns: var(--split-to); }

  /* animate only on real navigations, never on reload */
  .bb-animate .main-split { animation: split-in 1s ease both; }

  @keyframes split-in {
    from { grid-template-columns: var(--split-from); }
    to   { grid-template-columns: var(--split-to); }
  }
`;

const style = document.createElement("style");
style.textContent = css;
document.head.appendChild(style);

const navType = performance.getEntriesByType("navigation")[0]?.type;
if (navType !== "reload") {
  document.body.classList.add("bb-animate");
}