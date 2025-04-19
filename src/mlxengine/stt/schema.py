from enum import Enum
from typing import List, Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


class TimestampGranularity(str, Enum):
    WORD = "word"
    SEGMENT = "segment"


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class TranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: Optional[List[TranscriptionWord]] = None
    segments: Optional[List[Segment]] = None


class SimpleTranscriptionResponse(BaseModel):
    text: str


class STTRequestForm:
    def __init__(
        self,
        file: UploadFile = File(..., description="The audio file to transcribe"),
        model: str = Form(..., description="ID of the model to use"),
        language: Optional[str] = Form(
            None, description="The language of the input audio in ISO-639-1 format"
        ),
        prompt: Optional[str] = Form(
            None, description="An optional text to guide the model's style"
        ),
        response_format: Optional[ResponseFormat] = Form(
            ResponseFormat.JSON, description="The format of the transcription output"
        ),
        temperature: Optional[float] = Form(
            0.0, description="The sampling temperature, between 0 and 1"
        ),
        timestamp_granularities: Optional[List[str]] = Form(
            default=["segment"],
            alias="timestamp_granularities[]",  # 添加别名以匹配表单字段名
            description="The timestamp granularities to populate (word or segment)",
        ),
    ):
        self.file = file
        self.model = model
        self.language = language
        self.prompt = prompt
        self.response_format = response_format
        self.temperature = temperature

        # 转换 timestamp_granularities 字符串列表为枚举值列表
        self.timestamp_granularities = []
        if timestamp_granularities:
            for gran in timestamp_granularities:
                try:
                    self.timestamp_granularities.append(
                        TimestampGranularity(gran.lower())
                    )
                except ValueError:
                    raise ValueError(
                        f"Invalid timestamp granularity: {gran}. Must be one of: word, segment"
                    )
        else:
            self.timestamp_granularities = [TimestampGranularity.SEGMENT]

        self.validate()

    def validate(self):
        # Validate file extension
        allowed_extensions = {
            "flac",
            "mp3",
            "mp4",
            "mpeg",
            "mpga",
            "m4a",
            "ogg",
            "wav",
            "webm",
        }
        file_extension = self.file.filename.split(".")[-1].lower()
        if file_extension not in allowed_extensions:
            raise ValueError(
                f"File type not supported. Must be one of: {', '.join(allowed_extensions)}"
            )

        # Validate temperature
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")

        # Validate language code if provided
        if self.language and len(self.language) != 2:
            raise ValueError("Language code must be a 2-letter ISO-639-1 code")

        print(f"timestamp_granularities: {self.timestamp_granularities}")

        # 验证 word 时间戳必须使用 verbose_json
        if (
            self.timestamp_granularities
            and TimestampGranularity.WORD in self.timestamp_granularities
            and self.response_format != ResponseFormat.VERBOSE_JSON
        ):
            raise ValueError(
                "Word-level timestamps require response_format=verbose_json"
            )
