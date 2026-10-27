document.querySelectorAll(".tab").forEach(tab => tab.onclick = () => {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("is-active", t === tab));
  document.querySelectorAll(".tab-content").forEach(c => c.hidden = c.dataset.tab !== tab.dataset.tab);
});

const toggleTree = (row, open) => {
  document.querySelectorAll(`.tree-row[data-parent="${row.dataset.node}"]`).forEach(child => {
    child.hidden = !open;
    toggleTree(child, open);
  });
};

document.querySelectorAll(".tree-toggle").forEach(btn => btn.onclick = () => {
  const row = btn.closest(".tree-row");
  const open = btn.textContent === "\u25BE";
  btn.textContent = open ? "\u25B8" : "\u25BE";
  toggleTree(row, !open);
});

document.querySelectorAll(".tree-table").forEach(table => {
  const sum = [...table.tBodies[0].rows]
    .reduce((total, row) => total + Number(row.cells[1].dataset.weight), 0);
  table.querySelector("tfoot td").textContent = (Math.round(sum * 100) / 100 || 0).toFixed(2);
});