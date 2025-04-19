import json
import re
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlxengine.utils.logger import logger

from ...schema import (
    ChatMessage,
    FunctionCall,
    Role,
    SpecificToolChoice,
    Tool,
    ToolCall,
    ToolChoiceType,
)
from .chat_tokenizer import ChatTokenizer
from .utils import parse_tool_calls


class Llama3ChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "<|python_tag|>"
        self.end_tool_calls = ""
        self.strict_mode = False
        self.pre_fill_tools_prompt = ""

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def encode(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        **kwargs,
    ) -> str:
        prompt = super().encode(messages, tools, tool_choice, **kwargs)

        if tools:
            if isinstance(tool_choice, SpecificToolChoice):
                self.pre_fill_tools_prompt += self.start_tool_calls
                function_name = tool_choice.function["name"]

                self.pre_fill_tools_prompt += (
                    f"""{{"name": "{function_name}", "arguments":"""
                )

        return prompt + self.pre_fill_tools_prompt

    def _parse_strict_tools(self, text: str) -> Optional[List[ToolCall]]:
        tool_calls = []
        logger.debug(f"_parse_strict_tools: {text}")

        if text.strip().startswith(self.start_tool_calls):
            try:
                # Remove tool call tags and parse JSON directly
                json_str = text[len(self.start_tool_calls) :].strip()
                tool_data = json.loads(json_str)

                if isinstance(tool_data, dict) and "name" in tool_data:
                    # Get arguments and ensure they're a JSON string
                    args = tool_data.get("arguments", tool_data.get("parameters", {}))
                    if isinstance(args, str):
                        # Already a JSON string
                        arguments = args
                    else:
                        # Convert dict to JSON string
                        arguments = json.dumps(args)

                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            function=FunctionCall(
                                name=tool_data["name"],
                                arguments=arguments,
                            ),
                        )
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error parsing tool call: {e}")
                return None

        return tool_calls if tool_calls else None

    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse all tool calls from model output using regex.

        Llama 3 format often uses <|python_tag|> followed by a single JSON object.
        Example: `<|python_tag|>{"name": "tool1", ...} some text <|python_tag|>{"name": "tool2", ...}`

        Args:
            text: The model output text potentially containing multiple tool calls.

        Returns:
            ChatMessage: A message containing the aggregated parsed tool calls,
                         or the original text if no valid tool calls are found.
        """
        tool_calls = []
        response = self.pre_fill_tools_prompt + text

        # Regex to find "<|python_tag|>" followed by optional whitespace and then {...}
        # Captures the content inside the curly braces.
        # Assumes the tool call is a single JSON object, not an array like Mistral.
        pattern = re.compile(r"<\|python_tag\|>\s*(\{.*?\})", re.DOTALL)

        matches = pattern.finditer(response)

        for match in matches:
            json_str = match.group(1)  # Get the captured JSON object string "{...}"
            try:
                # Attempt to parse the captured string as a JSON object
                tool_data = json.loads(json_str)

                if not isinstance(tool_data, dict) or "name" not in tool_data:
                    print(f"Warning: Invalid tool call format (expected dict with 'name'): {tool_data}. Skipping.")
                    continue

                # Get arguments and ensure they're a JSON string
                args = tool_data.get("arguments", tool_data.get("parameters", {}))
                if isinstance(args, str):
                    arguments = args # Already a JSON string
                else:
                    arguments = json.dumps(args) # Convert dict to JSON string

                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        function=FunctionCall(
                            name=tool_data["name"],
                            arguments=arguments,
                        ),
                    )
                )
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON in Llama3 tool call block: {e} - JSON string: '{json_str}'")
                continue # Continue parsing other potential tool calls
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing Llama3 tool call data: {e} - Data: '{tool_data}'")
                continue # Continue processing other calls in the block or next match

        # Return message with aggregated tool calls if any were found, otherwise return original text
        if tool_calls:
            # TODO: Handle interleaved text?
            return ChatMessage(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=tool_calls,
            )
        else:
            # No valid tool calls found, return the original text
            # Note: We used 'response' which includes pre_fill, return original 'text'
            return ChatMessage(
                role=Role.ASSISTANT,
                content=text,
                tool_calls=None,
            )
