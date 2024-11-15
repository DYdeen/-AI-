import os
from dashscope import Generation

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '你是谁？'}
    ]
response = Generation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",   # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=messages,
    result_format="message"
)

if response.status_code == 200:
    print(response.output.choices[0].message.content)
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core.base.llms.types import MessageRole, ChatMessage

#DashScope 中的字段 "model_name" 与受保护的命名空间 "model_" 冲突
DashScope.model_config['protected_namespaces'] = ()#但是运行后也无法将警告取消。。。

# 初始化 DashScope 对象
dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key="sk-ccd3a2589b8b49ecb2218eb284906bb4")

# 存储对话历史的消息列表
messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
]

# 结束对话的关键词列表
end_keywords = ["再见", "结束对话", "谢谢", "拜拜","我知道了","感谢"]


def is_end_conversation(user_input):
    """检查用户输入是否包含结束对话的语义"""
    for keyword in end_keywords:
        if keyword in user_input:
            return True
    return False


# 主对话循环
while True:
    user_input = input("\n用户: ")  # 获取用户输入

    if is_end_conversation(user_input):
        print("系统: 感谢您的咨询，再见")
        break  # 结束对话

    # 将用户的输入添加到消息列表
    messages.append(ChatMessage(role=MessageRole.USER, content=user_input))

    # 调用chat接口进行回复（支持多轮对话）
    responses = dashscope_llm.stream_chat(messages)  # 使用流式输出
    print("系统： ",end="")
    for response in responses:
        print(response.delta, end="")  # 输出流式生成的内容

    # 将系统的回复添加到消息列表中
    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response.delta))
