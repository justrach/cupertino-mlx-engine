import json
import sys
import re # Added for client-side regex parsing
import uuid # Added for generating tool call IDs client-side
from datetime import datetime, timedelta
from openai import OpenAI, APIError

# --- Configuration ---
# Choose the model you are running with mlxengine
# Ensure the tokenizer used by mlxengine matches the expected format
MODEL = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit" # Uses <tool_call> format
# MODEL = "mlx-community/Llama-3.1-8B-Instruct-4bit" # Uses <|python_tag|> format
# MODEL = "mlx-community/Mistral-Nemo-Instruct-2407-4bit" # Uses [TOOL_CALLS] format

BASE_URL = "http://localhost:10240/v1" # Your mlxengine server address
API_KEY = "not-needed" # Replace if your server requires one

# --- Tool Definitions (OpenAI format) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "find_order_by_name",
            "description": "Finds a customer's order ID based on their name. Call this first when a customer asks about their order but doesn't provide an order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "The full name of the customer.",
                    },
                },
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
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's unique order identifier, potentially obtained using find_order_by_name.",
                    },
                },
                "required": ["order_id"],
            },
        }
    }
]

# --- Mock Tool Implementations (Replace with your actual logic) ---
def find_order_by_name(customer_name: str) -> dict:
    """Simulates finding an order ID based on customer name."""
    print(f"\n--- Tool Call: find_order_by_name(customer_name='{customer_name}') ---", file=sys.stderr)
    # Basic validation and simulation
    if isinstance(customer_name, str) and " " in customer_name.strip() and len(customer_name.strip()) > 3:
        simulated_id = f"ORD-{customer_name.strip().split()[0][:3].upper()}{len(customer_name.strip()):02d}"
        print(f"  -> Found order ID: {simulated_id}", file=sys.stderr)
        return {"order_id": simulated_id}
    else:
        print(f"  -> No order found for name: '{customer_name}' (Input type: {type(customer_name)})", file=sys.stderr)
        return {"order_id": None, "message": f"Could not find an order associated with the name '{customer_name}'. Please verify the name."}

def get_delivery_date(order_id: str) -> dict:
    """Simulates fetching delivery date based on order ID."""
    print(f"\n--- Tool Call: get_delivery_date(order_id='{order_id}') ---", file=sys.stderr)
    if isinstance(order_id, str) and order_id.strip().startswith("ORD-"):
        estimated_delivery = datetime.now() + timedelta(days=3)
        result = {
            "order_id": order_id,
            "estimated_delivery_date": estimated_delivery.strftime('%Y-%m-%d')
        }
        print(f"  -> Estimated Delivery: {result['estimated_delivery_date']}", file=sys.stderr)
        return result
    else:
         print(f"  -> Invalid Order ID format: '{order_id}' (Input type: {type(order_id)})", file=sys.stderr)
         return {"error": f"Invalid or missing order_id provided: '{order_id}'."}

# --- Function Mapping ---
available_functions = {
    "find_order_by_name": find_order_by_name,
    "get_delivery_date": get_delivery_date,
}

# --- Main Chat Loop Setup ---
print("Starting interactive multi-tool chat.")
print(f"Model: {MODEL}")
print(f"Server: {BASE_URL}")
print("Example: 'When will my package arrive?'")
print("Type 'exit' or 'quit' to end.")
print("-" * 30)

# Initialize OpenAI client
try:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    client.timeout = 60.0 # seconds timeout for API calls
except Exception as e:
    print(f"\nError initializing OpenAI client: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize conversation history
messages = [
    {
        "role": "system",
        "content": """You are a helpful customer support assistant focused on order delivery dates.
Follow these steps precisely:
1. Greet the user. If they ask about their order/delivery without providing details, ask for their *full name*. Do not ask for the order ID.
2. When the user provides a name, use the `find_order_by_name` tool. Do not guess or assume the name is correct.
3. If `find_order_by_name` returns an `order_id`, immediately use the `get_delivery_date` tool with that specific ID. You might do this in the same response if the model allows, or wait for the result of the first tool and then call the second.
4. If `find_order_by_name` returns no `order_id` (null or missing), inform the user politely that the order could not be found for that name and ask them to verify the name or provide an order ID if they have one.
5. Relay the estimated delivery date from `get_delivery_date` clearly to the user.
6. If any tool call results in an error, inform the user about the issue based on the error message.
Focus only on fulfilling the request using the tools. Be concise. Respond naturally."""
    }
]

# --- Main Execution Block ---
while True:
    # 1. Get User Input
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting chat.")
            break
        if not user_input.strip(): # Ignore empty input
            continue
    except (EOFError, KeyboardInterrupt): # Handle Ctrl+D or Ctrl+C
         print("\nExiting chat.")
         break

    messages.append({"role": "user", "content": user_input})

    # --- Inner Loop for Potential Multi-Turn Tool Use ---
    while True: # Loop until we get a final text response from the assistant
        try:
            # 2. Call the Model (Streaming)
            print("Assistant: ", end="", flush=True)
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto", # Let model decide, or force with {"type": "function", "function": {"name": "my_function"}}
                stream=True,
                temperature=0.5 # Optional: Adjust creativity (0.0 to 1.0)
            )

            # 3. Process the Streamed Response
            response_role = None
            full_content_accumulated = ""
            tool_calls_aggregated = [] # Stores completed tool call dicts from stream
            current_tool_call_info = {} # Accumulates parts for each tool call index

            stream_finish_reason = None
            for chunk in stream:
                delta = chunk.choices[0].delta
                stream_finish_reason = chunk.choices[0].finish_reason # Capture latest finish reason

                if delta.role:
                    response_role = delta.role

                # Accumulate and print text content
                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_content_accumulated += delta.content

                # Accumulate tool call information chunk by chunk
                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        index = tool_call_chunk.index
                        # Initialize storage for this tool call index if needed
                        if index not in current_tool_call_info:
                            current_tool_call_info[index] = {
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }

                        # Update fields for the specific tool call
                        if tool_call_chunk.id:
                            current_tool_call_info[index]["id"] = tool_call_chunk.id
                        if tool_call_chunk.function:
                            if tool_call_chunk.function.name:
                                current_tool_call_info[index]["function"]["name"] += tool_call_chunk.function.name
                            if tool_call_chunk.function.arguments:
                                current_tool_call_info[index]["function"]["arguments"] += tool_call_chunk.function.arguments

            # After stream finishes, finalize tool calls if reason was tool_calls
            if stream_finish_reason == "tool_calls":
                for index in sorted(current_tool_call_info.keys()):
                    # Basic validation before adding
                    tc = current_tool_call_info[index]
                    if tc.get("id") and tc.get("function", {}).get("name"):
                        tool_calls_aggregated.append(tc)
                    else:
                        print(f"\nWarning: Incomplete tool call chunk detected at index {index}: {tc}", file=sys.stderr)

            print() # Ensure newline after assistant output/stream ends

            # 4. Client-Side Parsing Fallback (if stream didn't yield structured tool calls)
            if not tool_calls_aggregated and full_content_accumulated.strip():
                print("\n--- No explicit tool calls in stream, attempting client-side parse ---", file=sys.stderr)
                extracted_calls = []
                # Define patterns for expected raw text formats
                patterns = {
                    # Qwen / generic HF format
                    "huggingface": re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
                    # Llama 3 format
                    "llama3": re.compile(r"<\|python_tag\|>\s*(\{.*?\})", re.DOTALL),
                    # Mistral format (captures JSON array)
                    "mistral": re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL),
                }

                # Selectively try patterns based on model or expected output
                # Example: Prioritize HuggingFace format
                matched_formats = set()
                for fmt, pattern in patterns.items():
                    try:
                        matches = pattern.finditer(full_content_accumulated)
                        for match in matches:
                            json_str = match.group(1).strip() # Group 1 contains the JSON
                            try:
                                tool_data = json.loads(json_str)

                                # Handle Mistral's array vs others' single object
                                if fmt == "mistral":
                                    if isinstance(tool_data, list):
                                        calls_in_match = tool_data
                                    elif isinstance(tool_data, dict): # Handle single obj in list format
                                        calls_in_match = [tool_data]
                                    else:
                                        print(f"  Client-Parse {fmt.upper()} Warning: Expected list/dict in {json_str}", file=sys.stderr)
                                        continue
                                else: # HuggingFace, Llama3 expect single object
                                     if isinstance(tool_data, dict):
                                         calls_in_match = [tool_data]
                                     else:
                                         print(f"  Client-Parse {fmt.upper()} Warning: Expected dict in {json_str}", file=sys.stderr)
                                         continue

                                # Process valid calls from the match
                                for call_dict in calls_in_match:
                                     if isinstance(call_dict, dict) and "name" in call_dict:
                                         # Arguments should already be present in the dict
                                         extracted_calls.append({
                                            "id": f"call_{uuid.uuid4().hex[:12]}", # Generate ID
                                            "type": "function",
                                            "function": call_dict # Use the already parsed dict
                                         })
                                         print(f"  Client-Parse {fmt.upper()} Success: Found {call_dict.get('name')}", file=sys.stderr)
                                         matched_formats.add(fmt)
                                     else:
                                         print(f"  Client-Parse {fmt.upper()} Warning: Invalid format in item {call_dict}", file=sys.stderr)

                            except json.JSONDecodeError:
                                print(f"  Client-Parse {fmt.upper()} Error: Invalid JSON '{json_str}'", file=sys.stderr)
                    except Exception as regex_err:
                        print(f"  Regex Error for {fmt}: {regex_err}", file=sys.stderr)


                if extracted_calls:
                     print("--- Client-side parse successful, proceeding with tool execution ---", file=sys.stderr)
                     tool_calls_aggregated = extracted_calls # Use client-parsed calls
                     # Optionally clear the raw text if tools were parsed out?
                     # full_content_accumulated = None

            # 5. Add Assistant's Response to History
            assistant_message = {"role": response_role or "assistant"}
            # Prefer adding parsed tool calls over raw content if both exist
            if tool_calls_aggregated:
                 assistant_message["tool_calls"] = tool_calls_aggregated
                 # Decide if you want to keep raw text when tools are parsed
                 # if full_content_accumulated and not tool_calls_aggregated: # Only add content if no tools parsed
                 #    assistant_message["content"] = full_content_accumulated
            elif full_content_accumulated.strip(): # Add content only if no tools and content exists
                assistant_message["content"] = full_content_accumulated

            # Add message only if it has content or tool calls, and avoid duplicates
            if assistant_message.get("content") or assistant_message.get("tool_calls"):
                 if not messages or messages[-1] != assistant_message:
                     messages.append(assistant_message)


            # 6. Execute Tools if Any Were Called (from stream or client parse)
            if tool_calls_aggregated:
                print("\n--- Executing Tool Call(s) ---", file=sys.stderr)
                tool_responses = []
                for tool_call in tool_calls_aggregated:
                    # Safely get components
                    tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:12]}") # Ensure ID exists
                    function_info = tool_call.get("function", {})
                    function_name = function_info.get("name")
                    function_args_obj = function_info.get("arguments", {}) # Could be string or dict

                    response_content = "" # Default content for tool response

                    # Ensure arguments are a dict for calling the Python function
                    function_args = {}
                    if isinstance(function_args_obj, str):
                        print(f"  Attempting Call: {function_name}( Args: '{function_args_obj}' )", file=sys.stderr)
                        try:
                            function_args = json.loads(function_args_obj)
                            if not isinstance(function_args, dict): # Should parse to dict
                                raise ValueError("Arguments JSON did not yield a dictionary")
                        except (json.JSONDecodeError, ValueError) as json_e:
                            error_msg = f"Invalid/malformed JSON arguments from model: {function_args_obj} ({json_e})"
                            print(f"  Error Parsing Args: {error_msg}", file=sys.stderr)
                            response_content = json.dumps({"error": error_msg})
                            tool_responses.append({
                                "role": "tool", "tool_call_id": tool_call_id,
                                "name": function_name or "unknown_function", "content": response_content,
                            })
                            continue # Skip execution for this tool call
                    elif isinstance(function_args_obj, dict):
                         function_args = function_args_obj
                         print(f"  Attempting Call: {function_name}( Args: {json.dumps(function_args)} )", file=sys.stderr)
                    else: # Handle unexpected argument format
                         error_msg = f"Unexpected argument format received for {function_name}: {type(function_args_obj)}"
                         print(f"  Error Parsing Args: {error_msg}", file=sys.stderr)
                         response_content = json.dumps({"error": error_msg})
                         tool_responses.append({
                             "role": "tool", "tool_call_id": tool_call_id,
                             "name": function_name or "unknown_function", "content": response_content,
                         })
                         continue # Skip execution

                    # Select and Call the Actual Function
                    if function_name and function_name in available_functions:
                        function_to_call = available_functions[function_name]
                        try:
                            function_response = function_to_call(**function_args) # Call with parsed dict args
                            response_content = json.dumps(function_response)
                            print(f"  Execution Success: Result = {response_content}", file=sys.stderr)
                        except TypeError as type_err: # Catch argument mismatches
                             error_msg = f"Argument mismatch calling function '{function_name}': {type_err}"
                             print(f"  Execution Error: {error_msg}", file=sys.stderr)
                             response_content = json.dumps({"error": error_msg})
                        except Exception as func_e: # Catch other execution errors
                            error_msg = f"Error executing function '{function_name}': {func_e}"
                            print(f"  Execution Error: {error_msg}", file=sys.stderr)
                            response_content = json.dumps({"error": error_msg})
                    elif not function_name:
                         error_msg = "Function name missing in tool call structure."
                         print(f"  Execution Error: {error_msg}", file=sys.stderr)
                         response_content = json.dumps({"error": error_msg})
                    else: # Function name provided but not found in available_functions
                        error_msg = f"Function '{function_name}' is not available/defined."
                        print(f"  Execution Error: {error_msg}", file=sys.stderr)
                        response_content = json.dumps({"error": error_msg})

                    # Append the tool's response message
                    tool_responses.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name or "unknown_function", # Ensure name is present
                        "content": response_content,
                    })
                # --- End of tool execution loop ---

                # 7. Add Tool Responses to History and Continue Inner Loop
                # Avoid adding duplicates if the loop errored and restarted
                if not messages or messages[-len(tool_responses):] != tool_responses:
                     messages.extend(tool_responses)
                print("--- Resuming conversation with tool results ---", file=sys.stderr)
                continue # Go back to step 2 to call the model again with tool results

            else:
                # 8. No Tool Calls Made, Break Inner Loop
                # The assistant's final text response was already printed during streaming.
                break # Exit the inner while loop, wait for next user input

        # --- Error Handling for the API Call ---
        except APIError as e:
            print(f"\nAPI Error: {e.status_code} - {e.message}", file=sys.stderr)
            # Decide if you want to retry or break
            break # Break inner loop on API error, wait for next user input
        except Exception as e:
            print(f"\nAn unexpected error occurred during API call/processing: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            break # Break inner loop on other errors

# --- End of Outer Main Loop ---