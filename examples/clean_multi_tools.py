# clean_multi_tool_calling.py

import json
import sys
import re
import uuid
from datetime import datetime, timedelta
from openai import OpenAI, APIError
from typing import List, Dict, Any, Optional, Tuple, Generator

# --- Configuration ---
# Choose the model you are running with mlxengine
MODEL = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit" # Uses <tool_call> format
# MODEL = "mlx-community/Llama-3.1-8B-Instruct-4bit" # Uses <|python_tag|> format
# MODEL = "mlx-community/Mistral-Nemo-Instruct-2407-4bit" # Uses [TOOL_CALLS] format

BASE_URL = "http://localhost:10240/v1" # Your mlxengine server address
API_KEY = "not-needed" # Replace if your server requires one
CLIENT_TIMEOUT = 60.0 # Seconds timeout for API calls

# --- Tool Definitions ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "find_order_by_name",
            "description": "Finds a customer's order ID based on their name. Call this first when a customer asks about their order but doesn't provide an order ID.",
            "parameters": {
                "type": "object",
                "properties": {"customer_name": {"type": "string", "description": "The full name of the customer."}},
                "required": ["customer_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the estimated delivery date for a specific order ID. Only call this *after* you have obtained the order ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "The customer's unique order identifier."}},
                "required": ["order_id"],
            },
        }
    }
]

# --- Mock Tool Implementations ---
def find_order_by_name(customer_name: str) -> dict:
    """Simulates finding an order ID."""
    print(f"\n--- Tool Call: find_order_by_name(customer_name='{customer_name}') ---", file=sys.stderr)
    if isinstance(customer_name, str) and " " in customer_name.strip() and len(customer_name.strip()) > 3:
        simulated_id = f"ORD-{customer_name.strip().split()[0][:3].upper()}{len(customer_name.strip()):02d}"
        print(f"  -> Found order ID: {simulated_id}", file=sys.stderr)
        return {"order_id": simulated_id}
    else:
        print(f"  -> No order found for name: '{customer_name}' (Input type: {type(customer_name)})", file=sys.stderr)
        return {"order_id": None, "message": f"Could not find order for '{customer_name}'. Verify name."}

def get_delivery_date(order_id: str) -> dict:
    """Simulates fetching delivery date."""
    print(f"\n--- Tool Call: get_delivery_date(order_id='{order_id}') ---", file=sys.stderr)
    if isinstance(order_id, str) and order_id.strip().startswith("ORD-"):
        estimated_delivery = datetime.now() + timedelta(days=3)
        result = {"order_id": order_id, "estimated_delivery_date": estimated_delivery.strftime('%Y-%m-%d')}
        print(f"  -> Estimated Delivery: {result['estimated_delivery_date']}", file=sys.stderr)
        return result
    else:
         print(f"  -> Invalid Order ID format: '{order_id}' (Input type: {type(order_id)})", file=sys.stderr)
         return {"error": f"Invalid order_id format: '{order_id}'."}

# --- Function Mapping ---
available_functions = {
    "find_order_by_name": find_order_by_name,
    "get_delivery_date": get_delivery_date,
}

# --- Helper Functions ---

def process_stream(stream: Generator) -> Tuple[Optional[str], str, List[Dict[str, Any]], Optional[str]]:
    """
    Processes the streaming response from the OpenAI API.

    Args:
        stream: The generator returned by client.chat.completions.create with stream=True.

    Returns:
        A tuple containing:
        - response_role (str | None): The role of the assistant.
        - full_content (str): The accumulated text content.
        - aggregated_tool_calls (List[Dict]): List of structured tool calls received.
        - finish_reason (str | None): The reason the stream finished.
    """
    response_role = None
    full_content_accumulated = ""
    aggregated_tool_calls = []
    current_tool_call_info = {} # index -> accumulated parts
    stream_finish_reason = None

    print("Assistant: ", end="", flush=True) # Print prompt immediately

    for chunk in stream:
        delta = chunk.choices[0].delta
        stream_finish_reason = chunk.choices[0].finish_reason

        if delta.role:
            response_role = delta.role
        if delta.content:
            print(delta.content, end="", flush=True)
            full_content_accumulated += delta.content
        if delta.tool_calls:
            for tool_call_chunk in delta.tool_calls:
                index = tool_call_chunk.index
                if index not in current_tool_call_info:
                    current_tool_call_info[index] = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                if tool_call_chunk.id:
                    current_tool_call_info[index]["id"] = tool_call_chunk.id
                if tool_call_chunk.function:
                    if tool_call_chunk.function.name:
                        current_tool_call_info[index]["function"]["name"] += tool_call_chunk.function.name
                    if tool_call_chunk.function.arguments:
                        current_tool_call_info[index]["function"]["arguments"] += tool_call_chunk.function.arguments

    # Finalize tool calls if stream indicated tool usage
    if stream_finish_reason == "tool_calls":
        for index in sorted(current_tool_call_info.keys()):
            tc = current_tool_call_info[index]
            if tc.get("id") and tc.get("function", {}).get("name"):
                aggregated_tool_calls.append(tc)
            else:
                 print(f"\nWarning: Incomplete tool call chunk @ index {index}: {tc}", file=sys.stderr)

    print() # Newline after stream
    return response_role, full_content_accumulated, aggregated_tool_calls, stream_finish_reason


def parse_raw_text_for_tools(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempts to parse tool calls from raw text if not provided by the stream.

    Args:
        text: The accumulated raw text output from the model.

    Returns:
        A list of parsed tool call dictionaries, or None if none found/parsed.
    """
    if not text or not text.strip():
        return None

    print("\n--- No explicit tool calls in stream, attempting client-side parse ---", file=sys.stderr)
    extracted_calls = []
    patterns = {
        "huggingface": re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
        "llama3": re.compile(r"<\|python_tag\|>\s*(\{.*?\})", re.DOTALL),
        "mistral": re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL),
    }

    for fmt, pattern in patterns.items():
        try:
            matches = pattern.finditer(text)
            for match in matches:
                json_str = match.group(1).strip()
                try:
                    tool_data = json.loads(json_str)
                    calls_in_match = []
                    if fmt == "mistral": # Mistral can send a list
                        calls_in_match = tool_data if isinstance(tool_data, list) else [tool_data]
                    else: # Others usually send a single object
                        calls_in_match = [tool_data] if isinstance(tool_data, dict) else []

                    for call_dict in calls_in_match:
                         if isinstance(call_dict, dict) and "name" in call_dict:
                             extracted_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": call_dict
                             })
                             print(f"  Client-Parse {fmt.upper()} Success: Found {call_dict.get('name')}", file=sys.stderr)
                         else:
                             print(f"  Client-Parse {fmt.upper()} Warning: Invalid format {call_dict}", file=sys.stderr)

                except json.JSONDecodeError:
                    print(f"  Client-Parse {fmt.upper()} Error: Invalid JSON '{json_str}'", file=sys.stderr)
        except Exception as regex_err:
             print(f"  Regex Error for {fmt}: {regex_err}", file=sys.stderr)

    if extracted_calls:
        print("--- Client-side parse successful ---", file=sys.stderr)
        return extracted_calls
    else:
        print("--- Client-side parse found no tool calls ---", file=sys.stderr)
        return None


def execute_tool_calls(
    tool_calls_to_execute: List[Dict[str, Any]],
    function_map: Dict[str, callable]
) -> List[Dict[str, Any]]:
    """
    Executes the requested tool calls and returns their responses.

    Args:
        tool_calls_to_execute: List of tool call dictionaries from the API/parsing.
        function_map: Dictionary mapping function names to callable functions.

    Returns:
        A list of tool response dictionaries for appending to the message history.
    """
    if not tool_calls_to_execute:
        return []

    print("\n--- Executing Tool Call(s) ---", file=sys.stderr)
    tool_responses = []
    for tool_call in tool_calls_to_execute:
        tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:12]}")
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name")
        function_args_obj = function_info.get("arguments", {})
        response_content = ""
        function_args = {}

        # 1. Parse Arguments String to Dict
        if isinstance(function_args_obj, str):
            print(f"  Attempting Call: {function_name}( Args: '{function_args_obj}' )", file=sys.stderr)
            try:
                function_args = json.loads(function_args_obj)
                if not isinstance(function_args, dict):
                    raise ValueError("Args JSON not a dictionary")
            except (json.JSONDecodeError, ValueError) as json_e:
                error_msg = f"Invalid/malformed JSON arguments: {function_args_obj} ({json_e})"
                print(f"  Error Parsing Args: {error_msg}", file=sys.stderr)
                response_content = json.dumps({"error": error_msg})
                # Add error response immediately and go to next tool call
                tool_responses.append({"role": "tool", "tool_call_id": tool_call_id, "name": function_name or "unknown", "content": response_content})
                continue
        elif isinstance(function_args_obj, dict):
             function_args = function_args_obj
             print(f"  Attempting Call: {function_name}( Args: {json.dumps(function_args)} )", file=sys.stderr)
        else:
             error_msg = f"Unexpected argument format: {type(function_args_obj)}"
             print(f"  Error Parsing Args: {error_msg}", file=sys.stderr)
             response_content = json.dumps({"error": error_msg})
             tool_responses.append({"role": "tool", "tool_call_id": tool_call_id, "name": function_name or "unknown", "content": response_content})
             continue

        # 2. Execute Function
        if function_name and function_name in function_map:
            function_to_call = function_map[function_name]
            try:
                function_response = function_to_call(**function_args)
                response_content = json.dumps(function_response)
                print(f"  Execution Success: Result = {response_content}", file=sys.stderr)
            except TypeError as type_err:
                 error_msg = f"Argument mismatch calling '{function_name}': {type_err}"
                 print(f"  Execution Error: {error_msg}", file=sys.stderr)
                 response_content = json.dumps({"error": error_msg})
            except Exception as func_e:
                error_msg = f"Error executing '{function_name}': {func_e}"
                print(f"  Execution Error: {error_msg}", file=sys.stderr)
                response_content = json.dumps({"error": error_msg})
        elif not function_name:
             error_msg = "Function name missing in tool call."
             print(f"  Execution Error: {error_msg}", file=sys.stderr)
             response_content = json.dumps({"error": error_msg})
        else:
            error_msg = f"Function '{function_name}' not available."
            print(f"  Execution Error: {error_msg}", file=sys.stderr)
            response_content = json.dumps({"error": error_msg})

        # 3. Append Formatted Tool Response
        tool_responses.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_name or "unknown_function",
            "content": response_content,
        })

    return tool_responses


def append_message(history: List[Dict[str, Any]], message: Dict[str, Any]):
    """Appends a message to history if it's valid and not a duplicate of the last one."""
    if not message or (not message.get("content") and not message.get("tool_calls")):
        return # Don't add empty messages
    if not history or history[-1] != message:
        history.append(message)


# --- Main Application ---
def main():
    print("Starting interactive multi-tool chat.")
    print(f"Model: {MODEL}")
    print(f"Server: {BASE_URL}")
    print("Example: 'When will my package arrive?'")
    print("Type 'exit' or 'quit' to end.")
    print("-" * 30)

    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=CLIENT_TIMEOUT)
    except Exception as e:
        print(f"\nError initializing OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)

    messages = [
        {
            "role": "system",
            "content": """You are a helpful customer support assistant focused on order delivery dates.
Follow these steps precisely:
1. Greet the user. If they ask about their order/delivery without providing details, ask for their *full name*. Do not ask for the order ID.
2. When the user provides a name, use the `find_order_by_name` tool. Do not guess or assume the name is correct.
3. If `find_order_by_name` returns an `order_id`, immediately use the `get_delivery_date` tool with that specific ID.
4. If `find_order_by_name` returns no `order_id` (null or missing), inform the user politely that the order could not be found and ask them to verify the name or provide an order ID if they have one.
5. Relay the estimated delivery date from `get_delivery_date` clearly to the user.
6. If any tool call results in an error, inform the user about the issue based on the error message.
Focus only on fulfilling the request using the tools. Be concise. Respond naturally."""
        }
    ]

    while True:
        # 1. Get User Input
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: break
            if not user_input.strip(): continue
        except (EOFError, KeyboardInterrupt): break

        append_message(messages, {"role": "user", "content": user_input})

        # --- Inner Interaction Loop ---
        while True:
            try:
                # 2. Call Model
                stream = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools,
                    tool_choice="auto", stream=True, temperature=0.5
                )

                # 3. Process Stream
                role, content, tool_calls, finish_reason = process_stream(stream)

                # 4. Try Client-Side Parsing if needed
                final_tool_calls = tool_calls # Start with tool calls from stream
                if not final_tool_calls and content:
                    parsed_calls = parse_raw_text_for_tools(content)
                    if parsed_calls:
                        final_tool_calls = parsed_calls
                        # Decide if content should be cleared if tools were parsed client-side
                        # content = None # Optional: Remove raw text if tools extracted

                # 5. Prepare and Append Assistant Message
                assistant_message = {"role": role or "assistant"}
                if final_tool_calls:
                    assistant_message["tool_calls"] = final_tool_calls
                # Only add content if no tools were identified or if you choose to keep it
                if content and not final_tool_calls:
                    assistant_message["content"] = content
                elif content and final_tool_calls:
                     # print("\n[Debug: Keeping content alongside parsed tools]", file=sys.stderr)
                     assistant_message["content"] = content # Keep content if desired

                append_message(messages, assistant_message)

                # 6. Execute Tools if Present
                if final_tool_calls:
                    tool_responses = execute_tool_calls(final_tool_calls, available_functions)
                    # Append tool responses without duplicate check (they are always new info)
                    messages.extend(tool_responses)
                    print("--- Resuming conversation with tool results ---", file=sys.stderr)
                    continue # Re-enter inner loop to call model with tool results
                else:
                    # No tool calls, break inner loop and wait for user
                    break

            except APIError as e:
                print(f"\nAPI Error: {e.status_code} - {e.message}", file=sys.stderr)
                break # Break inner loop, wait for user
            except Exception as e:
                print(f"\nUnexpected Error during interaction: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                break # Break inner loop, wait for user

    print("\nExiting chat.")

if __name__ == "__main__":
    main()