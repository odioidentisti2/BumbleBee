import asyncio
import html
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import bumblebee


BASE_DIR = Path(__file__).resolve().parent


log_queue: asyncio.Queue[str] = asyncio.Queue()


class QueueHandler(logging.Handler):
    """Feed the SSE log stream without losing records.

    Records are formatted on the emitting thread, then handed to the stream
    through `loop.call_soon_threadsafe` (training logs arrive on a worker
    thread). They are only removed from the queue once the streaming response
    is ready to send them, so a client disconnecting mid-stream cannot swallow
    them: they stay queued for the next connection.
    """

    def __init__(self, log_queue: asyncio.Queue[str]):
        super().__init__()
        self.setFormatter(logging.Formatter("%(message)s"))
        self.log_queue = log_queue
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        if self.loop is None:  # Startup not reached yet: no consumer, no loop needed
            self.log_queue.put_nowait(message)
        else:
            self.loop.call_soon_threadsafe(self.log_queue.put_nowait, message)


queue_handler = QueueHandler(log_queue)
queue_handler.setLevel(logging.INFO)  # DEBUG stays on the terminal only
logging.getLogger().addHandler(queue_handler)


@asynccontextmanager
async def lifespan(_: FastAPI):
    queue_handler.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="Tox-BumbleBee", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


@dataclass
class AppSession:
    model: object = None

session = AppSession()


DEVICE_NAME = bumblebee.init_device()


def arrival_path(request: Request) -> str:
    return urlparse(request.headers.get("referer", "")).path.rstrip("/") or "/"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request,
         "device": DEVICE_NAME},
    )

@app.get("/train", response_class=HTMLResponse)
def train(request: Request):
    return templates.TemplateResponse(
        "train.html",
        {"request": request,
         "device": DEVICE_NAME,
         "source": arrival_path(request)},
    )

@app.get("/test", response_class=HTMLResponse)
def test(request: Request):
    return templates.TemplateResponse(
        "test.html",
        {"request": request,
         "device": DEVICE_NAME,
         "source": arrival_path(request)},
    )

# TEMP FIX!
def with_dataset_path(dataset_info: dict) -> dict:
    dataset_info["path"] = str(BASE_DIR / "DATASETS" / dataset_info["file_name"])
    return dataset_info

@app.post("/api/train")
def run_training(dataset_info: dict):
    load_and_train(with_dataset_path(dataset_info))
    return {"success": True, "message": "Training completed"}

@app.post("/api/test")
def run_testing(payload: dict):
    model_file_name = payload.pop("model_file_name", None)
    if model_file_name:
        model_path = BASE_DIR / "MODELS" / Path(model_file_name).name
        if model_path.suffix.lower() != ".pt" or not model_path.is_file():
            raise HTTPException(status_code=400, detail=f"Model not found: {model_path.name}")
        session.model = bumblebee.load(str(model_path))
    if session.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    load_and_test(with_dataset_path(payload))
    return {"success": True}

# Output console
@app.get("/api/logs")
async def stream_logs():
    async def event_stream():
        while True:
            message = await log_queue.get()
            for line in message.splitlines() or [""]:
                yield f"data: {html.escape(line)}\ndata: \n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")



def load_and_train(trainingset_info):
    trainingset = bumblebee.load_dataset(trainingset_info, kind="Training")
    session.model = bumblebee.train(trainingset)


def load_and_test(testset_info):
    testset = bumblebee.load_dataset(testset_info, kind="Test")
    bumblebee.test(testset, session.model)


# DEBUG ONLY!!!!
@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


if __name__ == "__main__":
    # threading.Timer(1.0, open_browser).start()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
