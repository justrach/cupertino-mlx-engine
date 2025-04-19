# MLX Engine

[![image](https://img.shields.io/pypi/v/mlxengine.svg)](https://pypi.python.org/pypi/mlxengine)

![alt text](docs/banner.png)

> **Note:** MLX Engine is a fork of [MLX Omni Server](https://github.com/madroidmaq/mlx-omni-server) by [@madroidmaq](https://github.com/madroidmaq), refactored to use TurboAPI and enhance maintainability.

MLX Engine is a local inference server powered by Apple's MLX framework, specifically designed for Apple Silicon (M-series) chips. It implements
OpenAI-compatible API endpoints, enabling seamless integration with existing OpenAI SDK clients while leveraging the power of local ML inference.

## Features

- 🚀 **Apple Silicon Optimized**: Built on MLX framework, optimized for M1/M2/M3/M4 series chips
- 🔌 **OpenAI API Compatible**: Drop-in replacement for OpenAI API endpoints
- 🎯 **Multiple AI Capabilities**:
    - Audio Processing (TTS & STT)
    - Chat Completion
    - Image Generation
- ⚡ **High Performance**: Local inference with hardware acceleration
- 🔐 **Privacy-First**: All processing happens locally on your machine
- 🛠 **SDK Support**: Works with official OpenAI SDK and other compatible clients

## Supported API Endpoints

The server implements OpenAI-compatible endpoints:

- [Chat completions](https://platform.openai.com/docs/api-reference/chat): `/v1/chat/completions`
    - ✅ Chat
    - ✅ Tools, Function Calling
    - ✅ Structured Output
    - ✅ LogProbs
    - 🚧 Vision
- [Audio](https://platform.openai.com/docs/api-reference/audio)
    - ✅ `/v1/audio/speech` - Text-to-Speech
    - ✅ `/v1/audio/transcriptions` - Speech-to-Text
- [Models](https://platform.openai.com/docs/api-reference/models/list)
    - ✅ `/v1/models` - List models
    - ✅ `/v1/models/{model}` - Retrieve or Delete model
- [Images](https://platform.openai.com/docs/api-reference/images)
    - ✅ `/v1/images/generations` - Image generation

## Installation

```bash
# Install using pip
pip install mlxengine
```

## Quick Start

There are two ways to use MLX Engine:

### Method 1: Using the HTTP Server

1. Start the server:

```bash
# If installed via pip as a package
mlxengine
```

You can use `--port` to specify a different port, such as: `mlxengine --port 10240`. The default port is 10240.

You can view more startup parameters by using `mlxengine --help`.

2. Configure the OpenAI client to use your local server:

```python
from openai import OpenAI

# Configure client to use local server
client = OpenAI(
    base_url="http://localhost:10240/v1",  # Point to local server
    api_key="not-needed"  # API key is not required for local server
)
```

### Method 2: Using TestClient (No Server Required)

For development or testing, you can use TestClient to interact directly with the application without starting a server:

```python
from openai import OpenAI
from fastapi.testclient import TestClient # TODO: Update this import once TurboAPI has TestClient
from mlxengine.main import app

# Use TestClient to interact directly with the application
client = OpenAI(
    http_client=TestClient(app)  # Use TestClient directly, no network service needed
)
```

### Example Usage

Regardless of which method you choose, you can use the client in the same way:

```python
# Chat Completion Example
chat_completion = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[
        {"role": "user", "content": "What can you do?"}
    ]
)

# Text-to-Speech Example
response = client.audio.speech.create(
    model="lucasnewman/f5-tts-mlx",
    input="Hello, welcome to MLX Engine!"
)

# Speech-to-Text Example
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="mlx-community/whisper-large-v3-turbo",
    file=audio_file
)

# Image Generation Example
image_response = client.images.generate(
    model="argmaxinc/mlx-FLUX.1-schnell",
    prompt="A serene landscape with mountains and a lake",
    n=1,
    size="512x512"
)

# Tool Calling Example
import json
from datetime import datetime
from openai import OpenAI

model = "mlx-community/QwQ-32B-4bit" # Make sure this model supports tool calling
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
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

# First API call: The model decides to use the tool
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)

response_message = completion.choices[0].message
print("Assistant Response (Tool Call):")
print(response_message)

# Check if the model wants to call a tool
if response_message.tool_calls:
    print("\nTool Calls Detected:")
    print(response_message.tool_calls)

    # Append the assistant's message (with tool calls) to the conversation
    messages.append(response_message)

    # --- Simulate executing the function and getting the result ---
    # In a real application, you would execute the function based on the name and arguments
    tool_call = response_message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    if function_name == "get_delivery_date":
        order_id = function_args.get("order_id")
        # Simulate fetching data
        delivery_date = datetime.now()
        function_response = {
            "order_id": order_id,
            "delivery_date": delivery_date.strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        # Handle other potential function calls if needed
        function_response = {"error": "Unknown function"}

    # Append the tool response message to the conversation
    function_call_result_message = {
        "role": "tool",
        "content": json.dumps(function_response),
        "tool_call_id": tool_call.id
    }
    messages.append(function_call_result_message)
    
    print("\nTool Response Message (Appended):")
    print(function_call_result_message)

    # Second API call: Send the tool response back to the model
    print("\nSending tool response back to model...")
    completion_with_tool_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
    
    final_assistant_message = completion_with_tool_response.choices[0].message
    print("\nFinal Assistant Response:")
    print(final_assistant_message)
else:
    print("\nAssistant Response (No Tool Call):")
    print(response_message.content)
```

You can view more examples in [examples](examples).

## Contributing

We welcome contributions! If you're interested in contributing to MLX Engine, please check out our [Development Guide](docs/development_guide.md)
for detailed information about:

- Setting up the development environment
- Running the server in development mode
- Contributing guidelines
- Testing and documentation

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- API design inspired by [OpenAI](https://openai.com)
- Uses [TurboAPI](https://github.com/rachpradhan/turboapi) for the server implementation
- Chat(text generation) by [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- Image generation by [diffusionkit](https://github.com/argmaxinc/DiffusionKit)
- Text-to-Speech by [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx)
- Speech-to-Text by [mlx-whisper](https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md)
- Forked from [MLX Omni Server](https://github.com/madroidmaq/mlx-omni-server) by [@madroidmaq](https://github.com/madroidmaq)

## Disclaimer

This project is not affiliated with or endorsed by OpenAI or Apple. It's an independent implementation that provides OpenAI-compatible APIs using
Apple's MLX framework.

## Star History 🌟

[![Star History Chart](https://api.star-history.com/svg?repos=rachpradhan/mlxengine&type=Date)](https://star-history.com/#rachpradhan/mlxengine&Date)
