from pathlib import Path
import asyncio
import queue
import sys

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
    result = load_and_train(dataset_info)
    return {"success": True, "result": result}


@app.get("/api/logs")
async def stream_logs():
    async def event_stream():
        while True:
            line = await asyncio.to_thread(stdout_queue.get)
            yield f"data: {line}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# @app.post("/api/test")
# def run_testing(config: dict):
#     result = test(config)
#     return {"success": True, "result": result}


def load_and_train(trainingset_info):
    # print(trainingset_info)  # DEBUG
    sys.stdout = StdoutTee(sys.stdout, stdout_queue)
    trainingset = bumblebee.load_dataset(trainingset_info, train=True, flag='Training set')
    model = bumblebee.train(trainingset)
    return model

# def test_model(testset_info):
#     testset = bumblebee.load_dataset(testset_info, train=False, flag='Test set')
#     bumblebee.test(testset)


class StdoutTee:
    def __init__(self, original_stream, log_queue):
        self.original_stream = original_stream
        self.log_queue = log_queue
        self._buffer = ""

    def write(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()
        self._buffer += text

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.log_queue.put(line)

        return len(text)

    def flush(self):
        if self._buffer:
            self.log_queue.put(self._buffer)
            self._buffer = ""
        self.original_stream.flush()

    def isatty(self):
        return self.original_stream.isatty()

    @property
    def encoding(self):
        return self.original_stream.encoding

    def fileno(self):
        return self.original_stream.fileno()


stdout_queue = queue.Queue()


if __name__ == "__main__":
    # threading.Timer(1.0, open_browser).start()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
