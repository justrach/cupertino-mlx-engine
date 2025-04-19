import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Literal

from satya import Model, Field


class ToolType(str, Enum):
    FUNCTION = "function"


class FunctionParameters(Model):
    type: Optional[str] = Field(default=None)
    properties: Optional[Dict[str, Any]] = Field(default=None)
    required: Optional[List[str]] = Field(default=None)


class Function(Model):
    name: str = Field(max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(default=None)
    parameters: Optional[FunctionParameters] = Field(default=None)


class Tool(Model):
    type: ToolType = Field(default=ToolType.FUNCTION)
    function: Function


class ToolChoice(str, Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class SpecificToolChoice(Model):
    type: ToolType = Field(default=ToolType.FUNCTION)
    function: Dict[str, str]


class FunctionCall(Model):
    """Function call details within a tool call."""

    name: str
    arguments: str  # JSON string of arguments


class ToolCall(Model):
    """Tool call from model output."""

    id: str
    type: ToolType = Field(default=ToolType.FUNCTION)
    function: FunctionCall

    @classmethod
    def from_llama_output(
        cls, name: str, parameters: Dict[str, Any], call_id: str
    ) -> "ToolCall":
        """Create a ToolCall instance from Llama model output format."""
        return cls(
            id=call_id,
            type=ToolType.FUNCTION,
            function=FunctionCall(
                name=name,
                arguments=json.dumps(parameters),  # Convert parameters to JSON string
            ),
        )


ToolChoiceType = Union[ToolChoice, SpecificToolChoice]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(Model):
    role: Role
    content: Optional[Union[str, List[Dict[str, str]]]] = Field(default=None)
    thought: Optional[str] = Field(default=None)  # Optional field for model's thought process
    name: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)


class ChatCompletionUsageDetails(Model):
    reasoning_tokens: int = Field(default=0)
    accepted_prediction_tokens: int = Field(default=0)
    rejected_prediction_tokens: int = Field(default=0)


class PromptTokensDetails(Model):
    """包含提示令牌的详细信息"""

    cached_tokens: int = Field(default=0)


class ChatCompletionUsage(Model):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[PromptTokensDetails] = Field(default=None)


class ChatCompletionChoice(Model):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)


class ChatCompletionChunkChoice(Model):
    index: int
    delta: ChatMessage
    finish_reason: Optional[str] = Field(default=None)
    logprobs: Optional[Any] = Field(default=None)


class ChatCompletionChunk(Model):
    id: str
    object: str = Field(default="chat.completion.chunk")
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = Field(default=None)
    usage: Optional[ChatCompletionUsage] = Field(default=None)


class ChatCompletionResponse(Model):
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = Field(default=None)


class StreamOptions(Model):
    include_usage: bool = Field(default=False)


class JsonSchemaFormat(Model):
    description: Optional[str] = Field(
        default=None, description="A description of what the response format is for"
    )
    name: str = Field(
        description="The name of the response format",
        pattern="^[a-zA-Z0-9_-]+$",
        max_length=64
    )
    schema_def: Optional[Dict[str, Any]] = Field(
        default=None, description="The schema for the response format"
    )
    strict: Optional[bool] = Field(
        default=False, description="Whether to enable strict schema adherence"
    )


class ResponseFormat(Model):
    type: Literal["text", "json_object", "json_schema"] = Field(
        description="The type of response format"
    )
    json_schema: Optional[JsonSchemaFormat] = Field(
        default=None, description="The JSON schema configuration when type is 'json_schema'"
    )


class ChatCompletionRequest(Model):
    # Standard OpenAI API fields
    model: str = Field(description="ID of the model to use")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, min_value=0.0, max_value=2.0)
    top_p: Optional[float] = Field(default=1.0, min_value=0.0, max_value=1.0)
    max_tokens: Optional[int] = Field(default=None)
    max_completion_tokens: Optional[int] = Field(default=None)
    stream: Optional[bool] = Field(default=False)
    stream_options: Optional[StreamOptions] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=0.0, min_value=-2.0, max_value=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, min_value=-2.0, max_value=2.0)
    logit_bias: Optional[Dict[str, float]] = Field(default=None)
    logprobs: Optional[bool] = Field(default=False)
    top_logprobs: Optional[int] = Field(
        default=None,
        min_value=0,
        max_value=20,
    )
    n: Optional[int] = Field(default=1, min_value=1, max_value=10)
    tools: Optional[List[Tool]] = Field(default=None)
    tool_choice: Optional[ToolChoiceType] = Field(default=None)
    response_format: Optional[ResponseFormat] = Field(default=None)

    # Custom parameter for thinking models
    thinkingModel: Optional[bool] = Field(default=False)

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields: Set[str] = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
            "tools",
            "tool_choice",
            "stream_options",
            "response_format",
            "thinkingModel",
        }
        all_fields = vars(self)
        return {k: v for k, v in all_fields.items() if k not in standard_fields}
