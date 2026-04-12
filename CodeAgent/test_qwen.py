from openai import OpenAI
import sys

# 初始化客户端
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="myhpcvllmqwen101",
)

try:
    response = client.chat.completions.create(
        model="qwen-35b",
        messages=[
            {"role": "system", "content": "你是一个精通 Python 的编程助手。"},
            {"role": "user", "content": "写一个简单的 Python 打印 Hello World。"}
        ],
        max_tokens=512,
        temperature=0.7,
        stream=False  # 先关闭流式输出进行测试
    )

    # 检查返回对象并打印
    if hasattr(response, 'choices'):
        print("--- 模型回答 ---")
        print(response.choices[0].message.content)
    else:
        print("返回结果异常，原始响应内容如下：")
        print(response)

except Exception as e:
    print(f"发生错误: {e}")
