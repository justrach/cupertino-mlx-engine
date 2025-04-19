import json
import re
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ....utils.logger import logger
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


class HuggingFaceChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models.
    https://huggingface.co/blog/unified-tool-use
    """

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "<tool_call>\n"
        self.end_tool_calls = "</tool_call>"
        self.strict_mode = False
        self.pre_fill_tools_prompt = ""

    def encode(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        **kwargs,
    ):
        prompt = super().encode(messages, tools, tool_choice, **kwargs)

        if tools:
            if isinstance(tool_choice, SpecificToolChoice):
                self.pre_fill_tools_prompt += self.start_tool_calls
                function_name = tool_choice.function["name"]

                self.pre_fill_tools_prompt += (
                    f"""{{"name": "{function_name}", "arguments":"""
                )

        return prompt + self.pre_fill_tools_prompt

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def _parse_strict_tools(self, text: str) -> Optional[List[ToolCall]]:
        tool_calls = []
        logger.debug(f"_parse_strict_tools: {text}")

        if (
            text.strip().startswith(self.start_tool_calls)
            and self.end_tool_calls in text
        ):
            try:
                # Remove tool call tags and parse JSON directly
                json_str = text[
                    len(self.start_tool_calls) : text.find(self.end_tool_calls)
                ].strip()
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

        HuggingFace format uses <tool_call>\n...\n</tool_call> blocks.
        Example: `<tool_call>\n{"name": "tool1", ...}\n</tool_call> some text <tool_call>\n{"name": "tool2", ...}\n</tool_call>`

        Args:
            text: The model output text potentially containing multiple tool calls.

        Returns:
            ChatMessage: A message containing the aggregated parsed tool calls,
                         or the original text if no valid tool calls are found.
        """
        tool_calls = []
        response = self.pre_fill_tools_prompt + text

        # Regex to find "<tool_call>\n" followed by content until "</tool_call>"
        # Captures the content between the tags (non-greedy).
        # Assumes the content is a single JSON object.
        pattern = re.compile(r"<tool_call>\n(.*?)</tool_call>", re.DOTALL)

        matches = pattern.finditer(response)

        for match in matches:
            # Extract the JSON string, stripping potential whitespace/newlines
            json_str = match.group(1).strip()
            try:
                # Attempt to parse the captured string as a JSON object
                tool_data = json.loads(json_str)

                if not isinstance(tool_data, dict) or "name" not in tool_data:
                    print(f"Warning: Invalid HF tool call format (expected dict with 'name'): {tool_data}. Skipping.")
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
                logger.error(f"Error parsing JSON in HF tool call block: {e} - JSON string: '{json_str}'")
                continue # Continue parsing other potential tool calls
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing HF tool call data: {e} - Data: '{tool_data}'")
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
