from fastapi import FastAPI, Response
from pydantic import BaseModel
from tts import silero_tts

app = FastAPI()


class TranscribeRequest(BaseModel):
    audio: bytes


class TTSRequest(BaseModel):
    text: str
    speaker: str
    file: str


@app.get("/")
async def hello():
    return {"hello": "from silero"}


@app.post("/tts")
async def tts(request: TTSRequest):
    return silero_tts(request)
