const promptNode = document.querySelector(".console-prompt");


// Helper for console feedback (blink and <br>)
function appendLine(parts, blankBefore = false) {
  if (promptNode.childNodes.length) {
    promptNode.appendChild(document.createElement("br"));
    if (blankBefore) {
      promptNode.appendChild(document.createElement("br"));
    }
  }
  parts.forEach((part) => {
    if (typeof part === "string") {
      promptNode.appendChild(document.createTextNode(part));
      return;
    }
    const span = document.createElement("span");
    span.className = "blink";
    span.textContent = part.blink;
    promptNode.appendChild(span);
  });
}


function refreshConsole({ dataset = null, model = null } = {}) {
  promptNode.textContent = "";

  const page = document.body.dataset.page;

  if (page === "train") {
    if (dataset) {
      appendLine([`Dataset: ${dataset.file_name}`]);
      appendLine([">  ", { blink: "TRAIN" }], blankBefore = true);
    } else {
      appendLine(["> Upload a ", { blink: "DATASET" }]);
    }
    return;
  }

  if (page !== "test") return;

  if (dataset) {
    appendLine([`Dataset: ${dataset.file_name}`]);
  }
  if (model) {
    appendLine([`Model: ${model.trained ? "trained" : model.file_name}`]);
  }

  if (dataset && model) {
    appendLine([">  ", { blink: "TEST" }], blankBefore = true);
    return;
  } else if (!dataset && !model) {
    appendLine(["> Upload a ", { blink: "DATASET and a MODEL" }]);
  } else if (!dataset) {
    appendLine(["> Upload a ", { blink: "DATASET" }], blankBefore = true);
  } else {
    appendLine(["> Upload a ", { blink: "MODEL" }], blankBefore = true);
  }
}

function clearConsole() {
  promptNode.textContent = "";
}

// Initial render
refreshConsole();