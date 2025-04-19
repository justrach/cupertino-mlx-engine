from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any # Added Dict, Any

from mlx_lm.tokenizer_utils import TokenizerWrapper

# Assuming these imports point to your satya.Model definitions
from ...schema import ChatMessage, Role, Tool, ToolCall, ToolChoice, ToolChoiceType


class ChatTokenizer(ABC):
    """Base class for tools handlers."""

    start_tool_calls: str
    end_tool_calls: str

    def __init__(self, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer

    def encode(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        **kwargs,
    ) -> str:
        """Encode tools and conversation into a prompt string.

        This is a common implementation that uses the tokenizer's chat template.
        Subclasses can override this if they need different behavior.
        """
        schema_tools = None
        if tools:
            # Use .dict() and manually exclude None
            schema_tools = []
            for tool in tools:
                # Assuming satya.Model has a .dict() method
                raw_tool_dict = tool.dict()
                # Manually filter out None values
                schema_tools.append({k: v for k, v in raw_tool_dict.items() if v is not None})


        # Determine if the last message needs prefilling based on its role
        should_prefill = messages[-1].role == Role.ASSISTANT

        conversation = []
        for message in messages:
            # Use .dict() and manually exclude None for each message
            # Assuming satya.Model has a .dict() method
            raw_msg_dict = message.dict()
            # Manually filter out None values
            msg_dict = {k: v for k, v in raw_msg_dict.items() if v is not None}

            # Handle potential list content (assuming structure from original code)
            content = msg_dict.get("content")
            if isinstance(content, list):
                # Join text parts, assuming a specific list structure
                # Ensure this structure matches what your ChatMessage content can hold
                msg_dict["content"] = "\n\n".join(
                    item.get("text", "") # Use .get() for safety
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            elif not isinstance(content, (str, type(None))):
                # Handle unexpected content types if necessary
                # For example, convert to string or raise an error
                msg_dict["content"] = str(content)

            conversation.append(msg_dict)

        # Apply the chat template using the processed conversation and tools
        if should_prefill:
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tools=schema_tools, # Pass the processed tools
                tokenize=False,
                continue_final_message=True, # Continue the assistant's message
                **kwargs,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tools=schema_tools, # Pass the processed tools
                tokenize=False,
                add_generation_prompt=True, # Add prompt for model generation
                **kwargs,
            )

        # Append start tool call marker if required by tool_choice
        if tools and tool_choice == ToolChoice.REQUIRED:
             # Ensure start_tool_calls is defined in subclasses or here
            if hasattr(self, 'start_tool_calls') and self.start_tool_calls:
                prompt += self.start_tool_calls
            else:
                # Handle case where start_tool_calls is not defined - maybe log a warning
                print("Warning: ToolChoice.REQUIRED specified but start_tool_calls not defined.")


        return prompt

    @abstractmethod
    def decode_stream(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output stream."""
        pass

    @abstractmethod
    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse final model output potentially containing tool calls."""
        pass
