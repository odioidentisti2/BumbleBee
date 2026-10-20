
runButton.disabled = True
prompt.textContent = ""
if document.body.dataset.page == 'train':
  if dataset is not None:
    prompt.textContent += f"Dataset: {dataset.file_name}"
    runButton.disabled = False
elif document.body.dataset.page == 'test':
  if dataset is not None:
    prompt.textContent += f"Dataset: {dataset.file_name}"
  if model is not None:
    if prompt.textContext:
      prompt.textContent += "\n"
    prompt.textContent += f"Model: {model.file_name}"
  if dataset is not None and model is not None:
    runButton.disabled = False
    return
  if prompt.textContext:
    prompt.textContent += "\n"
  if dataset is None and model is None:
    prompt.textContext += "> Upload a DATASET and a MODEL"
  elif dataset is None:
    prompt.textContext += "> Upload a DATASET"
  elif model is None:
    prompt.textContext += "> Upload a MODEL"

