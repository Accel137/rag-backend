from openai import OpenAI

client = OpenAI(
    api_key="sk-57c6fc0c039248389b24cf94f474f482",
    base_url="https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
)

resp = client.responses.create(
    model="qwen3.5-plus",
    input="hello"
)

print(resp)