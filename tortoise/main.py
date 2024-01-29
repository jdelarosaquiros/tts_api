from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, Body, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient
from typing import Annotated


from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_400_BAD_REQUEST,
)
from enum import Enum
from pydantic import BaseModel
import argparse
import os
import torch
import torchaudio
import uuid
from time import time
import base64
import uvicorn
import ssl


from api import TextToSpeech, MODELS_DIR

from utils.audio import load_voices
from utils.text import split_and_recombine_text

#    parser = argparse.ArgumentParser()
#    parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
#    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
#                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random#')
#    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='fast')
#    parser.add_argument('--use_deepspeed', type=str, help='Which voice preset to use.', default=False)
#    parser.add_argument('--kv_cache', type=bool, help='If you disable this please wait for a long a time to get the output', default=True)
#    parser.add_argument('--half', type=bool, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True)
#    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
#    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
#                                                      'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
#    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)
#    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
#    parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
#    parser.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
#                                                          'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)
# args = parser.parse_args()

OUTPUT_PATH = "results/"
USERS_PATH = "users/"


class Preset(str, Enum):
    ultra_fast = "ultra_fast"
    fast = "fast"
    standard = "standard"
    high_quality = "high_quality"


class TortoiseModelParameters(BaseModel):
    user_id: str
    voice_name: str
    preset: Preset = Preset.fast
    use_deepspeed: bool = False
    kv_cache: bool = True
    half: bool = True
    candidates: int = 1
    cvvp_amount: float = 0.0
    seed: bool | None = None


class TortoiseModel:
    parameters: TortoiseModelParameters
    tts: TextToSpeech
    voice_samples: list
    conditioning_latents: tuple[torch.Tensor, torch.Tensor]

    def __init__(self, parameters: TortoiseModelParameters) -> None:
        self.parameters = parameters

        if self.parameters.candidates > 3:
            self.parameters.candidates = 3

        elif self.parameters.candidates < 1:
            self.parameters.candidates = 1

        self.tts = TextToSpeech(
            models_dir=MODELS_DIR,
            use_deepspeed=self.parameters.use_deepspeed,
            kv_cache=self.parameters.kv_cache,
            half=self.parameters.half,
        )

        if "&" in self.parameters.voice_name:
            voice_sel = self.parameters.voice_name.split("&")
        else:
            voice_sel = [self.parameters.voice_name]

        self.voice_samples, self.conditioning_latents = load_voices(
            voice_sel, [os.path.join("users", self.parameters.user_id)]
        )

    def generate_audio(self, text) -> str:
        audio_files: list[str] = []

        start_time = time()
        texts = split_and_recombine_text(text)
        all_parts = []
        for j, text in enumerate(texts):
            gen, dbg_state = self.tts.tts_with_preset(
                text,
                k=self.parameters.candidates,
                voice_samples=self.voice_samples,
                conditioning_latents=self.conditioning_latents,
                preset=self.parameters.preset,
                use_deterministic_seed=self.parameters.seed,
                cvvp_amount=self.parameters.cvvp_amount,
                return_deterministic_state=True,
            )
            end_time = time()

            if isinstance(gen, list):
                audio = gen[0].squeeze(0).cpu()

            else:
                audio = gen.squeeze(0).cpu()

            # torchaudio.save(
            #     os.path.join(
            #         self.parameters.output_path,
            #         self.parameters.user_id,
            #         f"{self.parameters.voice_name}_audio.wav",
            #     ),
            #     audio,
            #     24000,
            # )

            print(
                "Time taken to generate the audio: ", end_time - start_time, "seconds"
            )
            print("RTF: ", (end_time - start_time) / (audio.shape[1] / 24000))
            all_parts.append(audio)
        full_audio = torch.cat(all_parts, dim=-1)

        os.makedirs(os.path.join(OUTPUT_PATH, self.parameters.user_id), exist_ok=True)

        audio_path = os.path.join(
            OUTPUT_PATH,
            self.parameters.user_id,
            f"{self.parameters.voice_name}_audio.wav",
        )
        torchaudio.save(
            audio_path,
            full_audio,
            24000,
        )

        return audio_path


fastapi_key = os.environ["FASTAPI_KEY"]
ssl_cert_path = os.environ["SSL_CERT_PATH"]
ssl_key_path = os.environ["SSL_KEY_PATH"]
ssl_password = os.environ["SSL_PASSWORD"]

job_queue: dict[str, list[str]] = {}
auto_id: int = 0  # TODO: Use a string instead of int for more efficient space usage
models: dict[str, TortoiseModel] = {}
api_key_header = APIKeyHeader(name="Authentication")
app = FastAPI()

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(ssl_cert_path, keyfile=ssl_key_path, password=ssl_password)

os.makedirs(USERS_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == fastapi_key:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key.",
    )


def generate_user() -> str:
    users: set[str] = set(os.listdir("users"))
    user_id = uuid.uuid1().hex

    while user_id in users:
        user_id = uuid.uuid1().hex

    os.makedirs(os.path.join("users/", user_id), exist_ok=True)

    return user_id


def check_if_user_exist(user_id: str) -> bool:
    users = os.listdir("users")

    return user_id in users


def check_if_voice_exist(user_id: str, voice_name: str) -> bool:
    voices = os.listdir(os.path.join("users", user_id))

    return voice_name in voices


def get_voice_id(user_id: str, voice_name: str) -> str:
    return voice_name + "_" + user_id


async def save_file(path: str, file: UploadFile):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, file.filename), "wb") as disk_file:
        file_bytes = await file.read()

        disk_file.write(file_bytes)


def encode_audio(audio_file):
    with open(audio_file, "rb") as f:
        encoded_content = base64.b64encode(f.read())
    return encoded_content.decode("ascii")


def initialize_model(model_parameters: TortoiseModelParameters):
    voice_id = get_voice_id(model_parameters.user_id, model_parameters.voice_name)
    models[voice_id] = TortoiseModel(model_parameters)
    job_queue[voice_id] = []


def wait_job_queue(voice_id: str):
    job_id: str = uuid.uuid1().hex

    while job_id in job_queue[voice_id]:
        job_id = uuid.uuid1().hex

    job_queue[voice_id].append(job_id)

    while job_id != job_queue[voice_id][0]:
        continue

    job_queue[voice_id].pop(0)


@app.get("/")
async def root():
    return {"message": "Test Successfull!"}


@app.get("/file/{voice_id}")
async def get_file(voice_id: str, api_key: str = Security(get_api_key)):
    # path = models[model_id].parameters.voice
    return {"data": FileResponse(f"results/{voice_id}_audio.wav")}


@app.post("/create_voice")
async def create_voice(
    voice_name: str,
    sample_audio: UploadFile,
    user_id: str | None = None,
    api_key: str = Security(get_api_key),
):
    if not sample_audio.content_type.startswith("audio"):
        raise HTTPException(
            status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File provided is not an audio.",
        )

    if not user_id:
        user_id = generate_user()

    elif not len(user_id) > 20:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Provide longer unique id.",
        )

    # For more secure system, use this commented codes
    # elif not check_if_user_exist:
    #     raise HTTPException(
    #         status_code=HTTP_400_BAD_REQUEST,
    #         detail="User doesn't exists.",
    #     )

    user_path = os.path.join("users", user_id)
    voice_path = os.path.join(user_path, voice_name)

    # voices = os.listdir(user_path)
    # if voice_name in voices:
    #     raise HTTPException(
    #         status_code=HTTP_400_BAD_REQUEST,
    #         detail="Voice provided already exists.",
    #     )

    await save_file(voice_path, sample_audio)

    return {"user_id": user_id}


@app.post("/add_sample_audio")
async def add_sample_audio(
    user_id: str,
    voice_name: str,
    sample_audio: UploadFile,
    api_key: str = Security(get_api_key),
):
    if not sample_audio.content_type.startswith("audio"):
        raise HTTPException(
            status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File provided is not an audio.",
        )

    voice_path = os.path.join("users", user_id, voice_name)

    if not os.path.isdir(voice_path):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Voice does not exist.",
        )

    # Wrap it in a function called save_file()
    with open(os.path.join(voice_path, sample_audio.filename), "wb") as disk_file:
        file_bytes = await sample_audio.read()

        disk_file.write(file_bytes)

        print(
            f"Received file named {sample_audio.filename} containing {len(file_bytes)} bytes. "
        )

        # return FileResponse(disk_file.name, media_type=sample_audio.content_type)

    return {"status": "success"}


@app.post("/init_model")
async def init_model(
    model_parameters: TortoiseModelParameters, api_key: str = Security(get_api_key)
):
    if not check_if_user_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="User doesn't exists.",
        )

    if not check_if_voice_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Voice doesn't exists.",
        )

    initialize_model(model_parameters)
    return {"status": "success"}


# TODO: create a generate audio endpoint that uses a websocket, so it can return long audios without timeout errors
@app.post("/generate_audio")
async def generate_audio(
    text: Annotated[str, Body()],
    user_id: Annotated[str, Body()],
    voice_name: Annotated[str, Body()],
    api_key: str = Security(get_api_key),
):
    if not check_if_user_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="User doesn't exists.",
        )

    if not check_if_voice_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Voice doesn't exists.",
        )

    voice_id = get_voice_id(user_id, voice_name)

    if not voice_id in models:
        initialize_model(TortoiseModelParameters(voice_id=voice_id))

    wait_job_queue(voice_id)

    audio_path: str = models[voice_id].generate_audio(text)

    return {
        "completed": True,
        "data": encode_audio(audio_path),
    }


# TODO: create a generate audio endpoint that uses a websocket, so it can return long audios without timeout errors
@app.post("/generate_audio_test")
async def generate_audio_test(
    text: Annotated[str, Body()],
    user_id: Annotated[str, Body()],
    voice_name: Annotated[str, Body()],
    api_key: str = Security(get_api_key),
):
    if not check_if_user_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="User doesn't exists.",
        )

    if not check_if_voice_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Voice doesn't exists.",
        )

    voice_id = get_voice_id(user_id, voice_name)

    if not voice_id in models:
        initialize_model(TortoiseModelParameters(voice_id=voice_id))

    wait_job_queue(voice_id)

    audio_path: str = models[voice_id].generate_audio(text)

    return FileResponse(audio_path)


@app.websocket("/generate_long_audio")
async def generate_long_audio(
    websocket: WebSocket,
    text: Annotated[str, Body()],
    user_id: Annotated[str, Body()],
    voice_name: Annotated[str, Body()],
    api_key: str = Security(get_api_key),
):
    await websocket.accept()

    if not check_if_user_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="User doesn't exists.",
        )

    if not check_if_voice_exist:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Voice doesn't exists.",
        )

    voice_id = get_voice_id(user_id, voice_name)

    if not voice_id in models:
        initialize_model(TortoiseModelParameters(voice_id=voice_id))

    wait_job_queue(voice_id)

    audio_path: str = models[voice_id].generate_audio(text)

    await websocket.send_json(
        {
            "completed": True,
            "data": encode_audio(audio_path),
        }
    )


# def test_websocket():
#     client = TestClient(app)
#     expected = {}
#     with client.websocket_connect("/generate_long_audio") as websocket:
#         data: dict = websocket.receive_json()
#         print("here")
#         assert "data" in data.keys()
#         print(len(data["data"]))
#         wav_file = open("results/temp.wav", "wb")
#         decode_string = base64.b64decode(data["data"])
#         wav_file.write(decode_string)


# test_websocket()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_certfile=ssl_cert_path,
        ssl_keyfile=ssl_key_path,
        ssl_keyfile_password=ssl_password,
    )
