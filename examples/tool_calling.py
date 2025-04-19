import json
from datetime import datetime
from openai import OpenAI

model = "mlx-community/QwQ-32B-4bit"
client = OpenAI(
    base_url="http://localhost:10240/v1",  # Point to local server
    api_key="not-needed"  # API key is not required for local server
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
    }
]

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."
    },
    {
        "role": "user",
        "content": "Hi, can you tell me the delivery date for my order?"
    },
    {
        "role": "assistant", 
        "content": "Hi there! I can help with that. Can you please provide your order ID?"
    },
    {
        "role": "user", 
        "content": "i think it is order_12345"
    }
]

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)

response_message = completion.choices[0].message
print(response_message)
print(response_message.tool_calls)

messages.append(response_message)

order_id = "order_12345"
delivery_date = datetime.now()
tool_call_id = response_message.tool_calls[0].id

function_call_result_message = {
    "role": "tool",
    "content": json.dumps({
        "order_id": order_id,
        "delivery_date": delivery_date.strftime('%Y-%m-%d %H:%M:%S')
    }),
    "tool_call_id": tool_call_id
}
messages.append(function_call_result_message)

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)
print(completion.choices[0].message)