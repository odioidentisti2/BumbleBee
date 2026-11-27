function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];

    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }

    cur += ch;
  }

  out.push(cur);
  return out.map((value) => value.trim());
}

function parseCsvPreview(text, maxRows) {
  const lines = text
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);

  if (!lines.length) {
    return { headers: [], rows: [] };
  }

  const headers = parseCsvLine(lines[0]);
  const rows = [];

  for (let i = 1; i < lines.length && rows.length < maxRows; i += 1) {
    const raw = parseCsvLine(lines[i]);
    rows.push(headers.map((_, idx) => (typeof raw[idx] === "string" ? raw[idx] : "")));
  }

  return { headers, rows };
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// function makeDatasetKey() {
//   const page = (document.body.dataset.pageLabel || "dataset").toLowerCase();
//   return `bumblebee_dataset_config_${page}`;
// }

function buildModal() {
  const template = document.getElementById("csv-preview-template");
  if (!template) {
    return null;
  }

  const overlay = document.createElement("div");
  overlay.className = "csv-modal-overlay";
  overlay.innerHTML = template.innerHTML;

  document.body.appendChild(overlay);
  return overlay;
}

function initCsvPreviewModal() {
  if (!document.querySelector(".controls")) {
    return;
  }

  const overlay = buildModal();
  if (!overlay) {
    return;
  }

  const titleNode = overlay.querySelector(".csv-modal-title");
  const thead = overlay.querySelector(".csv-preview-table thead");
  const tbody = overlay.querySelector(".csv-preview-table tbody");
  const backBtn = overlay.querySelector(".csv-modal-back");
  const confirmBtn = overlay.querySelector(".csv-modal-confirm");
  const taskPanel = overlay.querySelector(".task-panel");
  const taskInputs = Array.from(overlay.querySelectorAll('input[name="prediction-task"]'));
  const toxMap = overlay.querySelector(".tox-map");
  const toxMapZero = overlay.querySelector(".tox-map-zero");
  const toxMapOne = overlay.querySelector(".tox-map-one");
  const toxMapGradient = overlay.querySelector(".tox-map-gradient");
  const toxMapToggle = overlay.querySelector(".tox-map-toggle");
  const stepNodes = {
    smiles: overlay.querySelector('[data-step-name="smiles"]'),
    target: overlay.querySelector('[data-step-name="target"]'),
    cas: overlay.querySelector('[data-step-name="cas"]'),
    task: overlay.querySelector('[data-step-name="task"]'),
  };

  const state = {
    file: null,
    headers: [],
    rows: [],
    activeStep: 0,
    selections: {
      smiles: -1,
      target: -1,
      cas: -1,
    },
    task: null,
    toxMap: null,
    toxSwapped: false,
  };

  function openModal() {
    overlay.classList.add("is-open");
  }

  function closeModal() {
    overlay.classList.remove("is-open");
  }

  function activeStepName() {
    if (state.activeStep === 0) {
      return "smiles";
    }
    if (state.activeStep === 1) {
      return "target";
    }
    return "cas";
  }

  function isHeaderSelected(idx) {
    return Object.values(state.selections).includes(idx);
  }

  function updateSteps() {
    ["smiles", "target", "cas", "task"].forEach((stepName, idx) => {
      const node = stepNodes[stepName];
      const valueNode = node.querySelector(".csv-selection-value");

      node.classList.remove("is-active", "is-inactive", "is-complete");
      if (idx < state.activeStep) {
        node.classList.add("is-complete");
      } else if (state.activeStep === 2 && (stepName === "cas" || stepName === "task")) {
        node.classList.add("is-active");
      } else if (idx === state.activeStep) {
        node.classList.add("is-active");
      } else {
        node.classList.add("is-inactive");
      }

      if (valueNode) {
        const headerIndex = state.selections[stepName];
        valueNode.textContent = headerIndex >= 0 ? state.headers[headerIndex] : "";
      }
    });
  }

  function updateButtons() {
    backBtn.disabled = state.activeStep === 0;
    confirmBtn.disabled = state.activeStep !== 2 || !state.task;
  }

  function updateTaskControls() {
    const isActive = state.activeStep === 2;

    taskPanel.classList.toggle("is-active", isActive);
    taskPanel.classList.toggle("is-inactive", !isActive);

    taskInputs.forEach((input) => {
      input.disabled = !isActive;
      input.checked = input.value === state.task;
    });

    const showToxMap = isActive && state.task === "binary_classification" && Boolean(state.toxMap);
    toxMap.classList.toggle("is-visible", showToxMap);
    toxMap.classList.toggle("is-inactive", !showToxMap);
    toxMapToggle.disabled = !showToxMap;

    if (showToxMap) {
      renderToxMap();
    }
  }

  function scanTargetClasses(targetIndex) {
    const labels = new Set();
    state.rows.forEach((row) => {
      const value = String(row[targetIndex] ?? "").trim();
      if (value) {
        labels.add(value);
      }
    });
    return [...labels].slice(0, 2);
  }

  function buildToxMap(targetIndex) {
    if (targetIndex < 0) {
      return null;
    }

    const [classA, classB] = scanTargetClasses(targetIndex);
    if (!classA || !classB) {
      return null;
    }

    return { classA, classB };
  }

  function orderedToxClasses() {
    const map = state.toxMap;
    return state.toxSwapped ? [map.classA, map.classB] : [map.classB, map.classA];
  }

  function renderToxMap() {
    const [zeroClass, oneClass] = orderedToxClasses();
    toxMapZero.textContent = zeroClass;
    toxMapOne.textContent = oneClass;
    toxMapToggle.checked = state.toxSwapped;
    toxMapGradient.classList.toggle("is-mirrored", state.toxSwapped);
  }

  function currentToxMap() {
    if (!state.toxMap) {
      return null;
    }

    const [zeroClass, oneClass] = orderedToxClasses();
    return { [zeroClass]: 0, [oneClass]: 1 };
  }

  function renderTable() {
    const headerHtml = state.headers
      .map((header, idx) => {
        const selectedClass = isHeaderSelected(idx) ? "is-selected" : "";
        const lockedAttr = isHeaderSelected(idx) ? ' data-locked="true"' : "";
        return `<th class="${selectedClass}" data-col-index="${idx}"${lockedAttr}>${escapeHtml(header)}</th>`;
      })
      .join("");

    thead.innerHTML = `<tr>${headerHtml}</tr>`;
    tbody.innerHTML = state.rows
      .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`)
      .join("");

    updateSteps();
    updateTaskControls();
    updateButtons();
  }

  function resetState() {
    state.file = null;
    state.headers = [];
    state.rows = [];
    state.activeStep = 0;
    state.selections.smiles = -1;
    state.selections.target = -1;
    state.selections.cas = -1;
    state.task = null;
    state.toxMap = null;
    state.toxSwapped = false;
  }

  function clearStep(stepIndex) {
    if (stepIndex <= 0) {
      state.selections.smiles = -1;
      state.selections.target = -1;
      state.selections.cas = -1;
      state.task = null;
      state.toxMap = null;
      state.toxSwapped = false;
      return;
    }

    if (stepIndex === 1) {
      state.selections.target = -1;
      state.selections.cas = -1;
      state.task = null;
      state.toxMap = null;
      state.toxSwapped = false;
      return;
    }

    if (stepIndex === 2) {
      state.selections.cas = -1;
      state.task = null;
      state.toxMap = null;
      state.toxSwapped = false;
    }
  }

  thead.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const colIndex = target.getAttribute("data-col-index");
    if (colIndex === null) {
      return;
    }

    const idx = Number(colIndex);
    if (isHeaderSelected(idx)) {
      return;
    }

    const stepName = activeStepName();
    state.selections[stepName] = idx;

    if (stepName === "smiles") {
      state.activeStep = 1;
    } else if (stepName === "target") {
      state.toxMap = buildToxMap(idx);
      state.toxSwapped = false;
      state.activeStep = 2;
    }

    renderTable();
  });

  backBtn.addEventListener("click", () => {
    if (state.activeStep === 0) {
      return;
    }

    state.activeStep -= 1;
    clearStep(state.activeStep);
    renderTable();
  });

  confirmBtn.addEventListener("click", () => {
    closeModal();

    document.dispatchEvent(new CustomEvent("dataset:selected", {
      detail: {
        task: state.task,
        file_name: state.file.name,
        smiles_header: state.headers[state.selections.smiles] || "",
        target_header: state.headers[state.selections.target] || "",
        id_header: state.selections.cas >= 0 ? state.headers[state.selections.cas] : "",
        tox_map: state.task === "binary_classification" ? currentToxMap() : null,
      },
    }));
   });

  taskInputs.forEach((input) => {
    input.addEventListener("change", () => {
      if (input.disabled || !input.checked) {
        return;
      }

      state.task = input.value;
      updateTaskControls();
      updateButtons();
    });
  });

  toxMapToggle.addEventListener("change", () => {
    if (!state.toxMap || state.task !== "binary_classification") {
      return;
    }

    state.toxSwapped = toxMapToggle.checked;
    renderToxMap();
  });

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) {
      closeModal();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && overlay.classList.contains("is-open")) {
      closeModal();
    }
  });


  function openPreviewForFile(file) {
    if (!(file instanceof File)) {
      return;
    }

    resetState();
    state.file = file;
    titleNode.textContent = `Preview: ${file.name}`;

    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === "string" ? reader.result : "";
      const parsed = parseCsvPreview(text, 5);
      state.headers = parsed.headers;
      state.rows = parsed.rows;
      renderTable();
      openModal();
    };

    reader.onerror = () => {
      closeModal();
    };

    reader.readAsText(file);
  }

  const fileInput = document.getElementById("control-input");
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const file = fileInput.files?.[0];
      if (file && file.name.toLowerCase().endsWith(".csv")) {
         openPreviewForFile(file);
      }
    });
  }
 }

 initCsvPreviewModal();