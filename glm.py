from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="……")  # 请填写您自己的 API Key

# 定义问题
question = """
Question: Give me the answer from 1 to 4: 
In the Netherlands, which of the following is an unusual common public practice? 
1. Tipping generously 
2. Using deodorant 
3. Talking loudly on the phone 
4. Cycle everywhere. 
This question is for a country or language that is Netherlands. You can only choose one option.
### Answer:
"""

# 创建对话请求
response = client.chat.completions.create(
    model="glm-4.7",  # 使用glm-4.7模型
    messages=[{"role": "user", "content": question}],  # 用户提问
    max_tokens=3,  # 增加max_tokens，确保模型有足够空间生成数字答案
    temperature=0.0,  # 设置温度为0，确保输出是确定的
    stop=["\n"]  # 使用stop参数，强制在换行符处停止输出
)

# 获取并打印完整回复
print(response)  # 打印整个response对象，查看实际结构
