import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
# 加载环境变量
load_dotenv()

def run_legacy_style(input):
    print("--- 方式一：传统直观写法 ---")
    
    # 1. 初始化模型
    model = init_chat_model(
        os.getenv("MODEL_NAME"),
        temperature=0.5,
        timeout=10,
        max_tokens=1000,
        api_key=os.getenv("API_KEY"),
        model_provider="ollama", # 这里需要安装 langchain-ollama 包
        base_url=os.getenv("BASE_URL"),
    )

    # 2. 定义 Prompt 模板
    # 注意：这里我们挖了三个“坑”：{field} {style} 和 {content}
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专注于 {field} 领域的翻译助手。请将用户的文本翻译成中文。\n"
             "翻译要求：必须带有 {style} 的语气，并保持专业性。"),
        ("user", "{content}")
    ])

    # 3. 渲染 Prompt (填坑)
    # 这一步，我们将变量填入，生成最终的消息列表 (List[Message])
    messages = template.invoke(input)
    
    print(f"[Debug] 渲染后的消息: {messages}")

    # 4. 调用模型
    response = model.invoke(messages)
    
    print(f"✅ 结果: {response.content}\n")


def test_legacy_style():
    input_data_1 = {
        "field": "软件工程",
        "style": "傲娇且略带讽刺",
        "content": "Using old-school monolithic architecture for a modern microservice problem is clearly an anti-pattern."
    }

    print("--- 场景一：傲娇的软件工程师 ---")
    print(f"输入文本: {input_data_1['content']}")
    
    run_legacy_style(input_data_1)


    print("\n" + "="*40 + "\n")
    
    # --- 6. 动态输入 2：历史文学领域，优雅语气 ---
    input_data_2 = {
        "field": "历史文学",
        "style": "优雅且充满哲理",
        "content": "The long river of time eventually reveals the true measure of a man's character."
    }
    
    print("--- 场景二：哲学的历史学者 ---")
    print(f"输入文本: {input_data_2['content']}")

    run_legacy_style(input_data_2)


from langchain_core.output_parsers import StrOutputParser

def run_lcel_style(input):
    print("--- 方式二：LCEL 链式写法 ---")
    
    # 1. 初始化模型 (同上)
    model = model = init_chat_model(
        os.getenv("MODEL_NAME"),
        temperature=0.5,
        timeout=10,
        max_tokens=1000,
        api_key=os.getenv("API_KEY"),
        model_provider="ollama", # 这里需要安装 langchain-ollama 包
        base_url=os.getenv("BASE_URL"),
    )

    # 2. 定义模板 (同上)
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专注于 {field} 领域的翻译助手。请将用户的文本翻译成中文。\n"
             "翻译要求：必须带有 {style} 的语气，并保持专业性。"),
        ("user", "{content}")
    ])

    # 3. 定义输出解析器 (可选)
    # 它能把 AI Message 对象直接转成纯字符串，省去我们手动取 .content
    parser = StrOutputParser()

    # 4. 🔗 组装链 (Chain)
    # 数据流向：字典输入 -> 模板渲染 -> 模型推理 -> 结果解析
    chain = template | model | parser

    # 5. 调用链
    # 直接传入字典，LCEL 会自动匹配模板中的变量
    result = chain.invoke(input)

    print(f"✅ 结果: {result}")

if __name__ == "__main__":
    # 如果想运行传统写法，请取消下面这行的注释
    # test_legacy_style()

    input_data_1 = {
        "field": "软件工程",
        "style": "傲娇且略带讽刺",
        "content": "Using old-school monolithic architecture for a modern microservice problem is clearly an anti-pattern."
    }

    run_lcel_style(input_data_1)