from fastapi import FastAPI, File, UploadFile, Response
from pydantic import BaseModel
from tts import speak

app = FastAPI()


class TranscribeRequest(BaseModel):
    audio: bytes


class TTSRequest(BaseModel):
    text: str
    speaker: str

@app.get("/")
async def hello():
    return {"hello": "from SAI"}


@app.post("/tts")
async def tts(request: TTSRequest):
    audio = speak(request.text, request.speaker)
    return Response(audio, media_type="audio/mpeg")