import json
import re
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...schema import ChatMessage, FunctionCall, Role, ToolCall
from .chat_tokenizer import ChatTokenizer


class MistralChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "[TOOL_CALLS]"
        self.end_tool_calls = ""

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse all tool calls from model output using regex.

        The model might output multiple function call blocks like:
        ][{"name": "tool1", ...}] some text ][{"name": "tool2", ...}]

        Args:
            text: The model output text potentially containing multiple tool calls.

        Returns:
            ChatMessage: A message containing the aggregated parsed tool calls,
                         or the original text if no valid tool calls are found.
        """
        tool_calls = []
        pattern = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)

        matches = pattern.finditer(text)

        for match in matches:
            json_str = match.group(1)
            try:
                tool_data = json.loads(json_str)

                if isinstance(tool_data, dict):
                    tool_data = [tool_data]
                elif not isinstance(tool_data, list):
                    print(f"Warning: Expected list inside ][, got {type(tool_data)}. Skipping.")
                    continue

                for call in tool_data:
                    if not isinstance(call, dict) or "name" not in call:
                        print(f"Warning: Invalid tool call item format: {call}. Skipping.")
                        continue

                    args = call.get("arguments", call.get("parameters", {}))
                    if isinstance(args, str):
                        arguments = args
                    else:
                        arguments = json.dumps(args)

                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            function=FunctionCall(
                                name=call["name"],
                                arguments=arguments,
                            ),
                        )
                    )
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in tool call block: {e} - JSON string: '{json_str}'")
                continue
            except (KeyError, ValueError) as e:
                print(f"Error processing tool call data: {e} - Data: '{tool_data}'")
                continue

        if tool_calls:
            return ChatMessage(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=tool_calls,
            )
        else:
            return ChatMessage(
                role=Role.ASSISTANT,
                content=text,
                tool_calls=None,
            )
