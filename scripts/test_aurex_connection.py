"""
Minimal test: verify LocalAi responds correctly to OpenAI Python SDK calls,
the same way AUREX will call it.
"""
from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="localai",
)

print("Test 1: List models")
models = client.models.list()
for model in models.data:
    print(f"  - {model.id}")

print("\nTest 2: Non-streaming completion")
response = client.chat.completions.create(
    model="qwen3.5-4b-q4_k_m",
    messages=[
        {"role": "system", "content": "Answer in one sentence only."},
        {"role": "user", "content": "What is LocalAi?"},
    ],
    max_tokens=100,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(f"  Response: {response.choices[0].message.content}")

print("\nTest 3: Streaming completion")
stream = client.chat.completions.create(
    model="qwen3.5-4b-q4_k_m",
    messages=[
        {"role": "user", "content": "Count to 3."},
    ],
    max_tokens=100,
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("  Streaming tokens: ", end="", flush=True)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n  Stream complete.")

print("\nAll tests passed. LocalAi is OpenAI SDK compatible.")
