"""
BumbleBee GUI — FastAPI + HTMX backend

Run with:  python app.py
Then open: http://localhost:8080
"""

import asyncio
import csv
import io
import threading
import traceback
from pathlib import Path
from typing import Generator
from queue import Queue, Empty

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

# Import ML training code
from bumblebee import main_loop
import reproducibility

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = FastAPI()
env = Environment(loader=FileSystemLoader("templates"))

# Serve static files (create static/ folder if needed)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

# Global log queue for streaming
log_queue: Queue = Queue()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def detect_task(values: list[str]) -> tuple[str, list[str]]:
    """Detect task type from sample values."""
    if not values:
        return "error", []

    # Try numeric
    try:
        [float(v) for v in values]
        return "regression", []
    except ValueError:
        pass

    # String labels
    unique = sorted(set(values))
    if len(unique) == 2:
        return "binary_classification", unique
    elif len(unique) > 2:
        return "multiclass", unique
    else:
        return "error", []


def build_dataset_info(
    csv_path: str,
    smiles_header: str,
    target_header: str,
    task: str,
    positive_label: str = None,
) -> dict:
    """Build the final dataset_info dict."""
    info = {
        "task": task,
        "path": csv_path,
        "smiles_header": smiles_header,
        "target_header": target_header,
    }
    if task == "binary_classification" and positive_label:
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                values = [row.get(target_header, '').strip() for row in reader if row.get(target_header)]
                unique = set(values)
                if len(unique) == 2:
                    other = [l for l in unique if l != positive_label][0]
                    info["tox_map"] = {positive_label: 1, other: 0}
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main page."""
    template = env.get_template("index.html")
    return template.render(device=str(device))


@app.post("/api/load-csv")
async def load_csv(file: UploadFile = File(...)):
    """Read CSV file and extract headers."""
    try:
        content = await file.read()
        text = content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(text))
        headers = list(reader.fieldnames or [])
        
        # Save file to temp location
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        temp_file.write(text)
        temp_file.close()
        
        return {
            "success": True,
            "headers": headers,
            "path": temp_file.name,
            "filename": file.filename,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/detect-task")
async def detect_task_endpoint(csv_path: str = Form(...), target_header: str = Form(...)):
    """Detect task type from target column."""
    try:
        values = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 200:
                    break
                v = row.get(target_header, '').strip()
                if v:
                    values.append(v)
        
        task, unique = detect_task(values)
        return {
            "success": True,
            "task": task,
            "unique_labels": unique,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/train")
async def start_training(
    csv_path: str = Form(...),
    csv_name: str = Form(default=""),
    smiles_header: str = Form(...),
    target_header: str = Form(...),
    task: str = Form(...),
    positive_label: str = Form(default=""),
):
    """Start training in background thread."""
    try:
        dataset_info = build_dataset_info(
            csv_path,
            smiles_header,
            target_header,
            task,
            positive_label,
        )
        
        if not all([dataset_info.get("task"), dataset_info.get("path"), 
                   dataset_info.get("smiles_header"), dataset_info.get("target_header")]):
            return {"success": False, "error": "Missing required fields"}
        
        if task == "binary_classification" and not positive_label:
            return {"success": False, "error": "Please select positive class"}
        
        # Clear queue and start thread
        while not log_queue.empty():
            log_queue.get_nowait()
        
        thread = threading.Thread(
            target=_run_training,
            args=(dataset_info,),
            daemon=True,
        )
        thread.start()
        
        return {"success": True, "message": "Training started"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/logs/stream")
async def stream_logs():
    """SSE endpoint for log streaming."""
    def log_generator() -> Generator[str, None, None]:
        while True:
            try:
                item = log_queue.get(timeout=1.0)
                if item is None:  # Training complete
                    yield f"data: __DONE__\n\n"
                    break
                # Keep original text, including newlines, so frontend rendering is faithful.
                safe = str(item).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                yield f"data: \"{safe}\"\n\n"
            except Empty:
                pass
    
    return StreamingResponse(log_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------

class _QueueWriter(io.TextIOBase):
    """Redirect stdout to queue."""
    def __init__(self, q: Queue):
        self._q = q
        self._buffer = ""

    def write(self, text: str):
        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._q.put(f"{line}\n")
        return len(text)

    def flush(self):
        if self._buffer:
            self._q.put(self._buffer)
            self._buffer = ""


def _run_training(dataset_info: dict):
    """Run training, capturing all output to queue."""
    import sys
    import time
    
    old_stdout, old_stderr = sys.stdout, sys.stderr
    writer = _QueueWriter(log_queue)
    sys.stdout = writer
    sys.stderr = writer
    
    try:
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"DEVICE: {device}")
        print()
        
        if dataset_info["task"] == "binary_classification":
            reproducibility.use_deterministic_algorithms(device)
        
        reproducibility.set_torch_seed()
        start = time.time()
        main_loop(dataset_info, device)
        print()
        print(f"TOTAL TIME: {time.time() - start:.0f}s")
        print()
        writer.flush()
        log_queue.put(None)
    except Exception:
        traceback.print_exc()
        writer.flush()
        log_queue.put(None)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser
    import time
    
    print(f"Starting BumbleBee on http://localhost:8080")
    print(f"Device: {device}")
    
    # Open browser in background
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8080")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")