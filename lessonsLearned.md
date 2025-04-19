# Lessons Learned

## Handling Streamed Tool Calls from Local LLMs

**Challenge:** When using local LLMs via frameworks like `mlxengine`, the server might stream the model's raw output token-by-token. If the model generates tool calls (e.g., `<tool_call>{...}</tool_call>`), this often arrives as simple text content within the stream, rather than structured `tool_calls` objects in the delta chunks like some hosted APIs provide. This prevents the client application from easily detecting the *intent* to call a tool mid-stream or via the standard `finish_reason="tool_calls"`.

**Initial Approach & Problem:** Modifying the server-side (`mlxengine`) tokenizer's `decode` method helps parse multiple tool calls from the *final, complete* response string. However, this doesn't solve the issue for *streaming* clients that expect structured tool call information within the stream itself. The client might finish processing the stream, see no structured `tool_calls`, and incorrectly assume the model just provided a text response containing raw tool call tags.

**Solution: Client-Side Parsing Fallback:**
To handle this robustly on the client-side:
1.  Process the stream normally, accumulating the full text content and any structured `tool_calls` deltas (if the server *does* send them).
2.  **After** the stream finishes, check if any structured `tool_calls` were received.
3.  If not, and if text content was received, use regular expressions to scan the *accumulated text content* for known tool call patterns (e.g., `<tool_call>{...}</tool_call>`, `[TOOL_CALLS][...]`, `<|python_tag|>{...}`).
4.  If patterns are found, parse the extracted JSON content to build the structured `tool_calls` list manually on the client.
5.  Proceed with the tool execution logic using this potentially client-parsed list.

**Key Considerations:**
*   **Regex Patterns:** Maintain specific regex patterns for different model families (Mistral, Llama 3, HuggingFace/Qwen, etc.) as their tool call syntax varies.
*   **Robust Parsing:** Handle potential JSON decoding errors gracefully during client-side parsing.
*   **Multi-Turn Flow:** Remember that tool calling is inherently multi-turn. The client *must* implement the full loop:
    *   Call API with user message + history.
    *   Receive response (text and/or tool calls).
    *   Detect/parse tool calls (structured or client-side fallback).
    *   If tools called: Execute them, get results.
    *   Append assistant's request + tool results to history.
    *   Call API *again* with updated history.
    *   Receive final text response.

This client-side fallback approach makes the application resilient to different server/model streaming behaviors regarding tool calls.

*(You can add more lessons here as the project evolves)* 