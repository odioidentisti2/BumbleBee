from pathlib import Path
import asyncio
import logging
import queue

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import bumblebee


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Tox-BumbleBee")

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


DEVICE = bumblebee.get_device()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request,
         "device": DEVICE},
    )

@app.get("/train", response_class=HTMLResponse)
def train(request: Request):
    return templates.TemplateResponse(
        "train.html",
        {"request": request,
         "device": DEVICE},
    )

@app.get("/test", response_class=HTMLResponse)
def test(request: Request):
    return templates.TemplateResponse(
        "test.html",
        {"request": request,
         "device": DEVICE},
    )


@app.post("/api/train")
def run_training(dataset_info: dict):
    dataset_info['path'] = 'DATASETS/' + dataset_info['file_name']  # TEMP FIX!
    result = load_and_train(dataset_info)
    return {"success": True, "result": result}


@app.get("/api/logs")
async def stream_logs():
    async def event_stream():
        while True:
            message = await asyncio.to_thread(stdout_queue.get)
            for line in message.splitlines() or [""]:
                yield f"data: {line}\n"
            yield "\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# @app.post("/api/test")
# def run_testing(config: dict):
#     result = test(config)
#     return {"success": True, "result": result}


def load_and_train(trainingset_info):
    trainingset = bumblebee.load_dataset(trainingset_info, train=True, flag='Training set')
    model = bumblebee.train(trainingset)
    return model

# def test_model(testset_info):
#     testset = bumblebee.load_dataset(testset_info, train=False, flag='Test set')
#     bumblebee.test(testset)


class QueueHandler(logging.Handler):
    def emit(self, record):
        try:
            self.log_queue.put(self.format(record))
        except Exception:
            self.handleError(record)

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue


stdout_queue = queue.Queue()
queue_handler = QueueHandler(stdout_queue)
queue_handler.setLevel(logging.INFO)
queue_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(queue_handler)


if __name__ == "__main__":
    # threading.Timer(1.0, open_browser).start()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
