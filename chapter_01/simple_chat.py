# 导入相关依赖
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os


# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

def test_translation():
    """
    实战：测试一个简单的翻译任务
    展示 LangChain 如何统一模型调用接口
    """
    

    # --- 1. 获取配置 ---
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    provider = os.getenv("MODEL_PROVIDER", "ollama")

    print(f"🚀 正在初始化模型: {model_name} ({provider})...")

    # --- 2. 初始化模型 (核心) ---
    # init_chat_model 是一个包装器，可以初始化不同厂商的模型。这是官方提供的一个统一接口，支持的模型厂商列表：https://docs.langchain.com/oss/python/integrations/chat
    # 也可以使用 ChatOpenAI 来初始化，需要安装 langchain-openai 包
    # 这里以 Ollama 为例，需要安装 langchain-ollama 包。 等同于使用 ChatOllama 初始化
    model = init_chat_model(
        model_name,
        temperature=0.5,
        timeout=10,
        max_tokens=1000,
        api_key=api_key,
        model_provider="ollama", # 这里需要安装 langchain-ollama 包
        base_url=base_url,
    )

    # --- 3. 构造消息 ---
    # LangChain 提供了消息类，用于表示不同角色的消息
    # message 对象有三个属性：
    # Role - 消息类型 (e.g. system, user)
    # Content - 消息内容 (like text, images, audio, documents, etc.)
    # Metadata - 可选字段，如响应信息、消息 ID、token 使用情况等
    
    # SystemMessage - 系统消息，用于设置模型的上下文
    system_msg = SystemMessage("你是一个精通【信达雅】翻译准则的助手。请将用户的输入翻译成中文，并附带一句简短的赏析。")
    
    # HumanMessage - 人类消息，用于设置模型的输入
    human_msg = HumanMessage("The only way to do great work is to love what you do.")

    
    # --- 4. 定义 Agent/Chain ---
    agent = create_agent(
        model=model,
        system_prompt=system_msg,
    )

    
   
    print(f"🔄 正在发送请求...")

    # --- 5. 调用模型 (Invoke) ---
    # 无论底层是哪个厂商，这里永远只用 .invoke()
    messages = [human_msg] # 在create_agent时已经包含了system_msg，所以这里只需要human_msg， 如果需要多个消息，可以在这里添加
    # 输入结构标准化为 {"messages": [...]}
    response = agent.invoke({"messages": messages})

    # --- 6. 打印结果 ---
    # response 是一个 AIMessage 对象，.content 才是文本内容
    # 响应的结构：https://docs.langchain.com/oss/python/api_reference/langchain.schema.messages
    # 例如：{'messages': [HumanMessage(content='The only way to do great work is to love what you do.', additional_kwargs={}, response_metadata={}, id='dced4b3e-016a-4f23-8ea2-b6c5c3a95343'), AIMessage(content='翻译：做出卓越成就的唯一途径是热爱你所做的事情。\n\n赏析：将"great work"译为"卓越成就"既保留了原意，又提升了语言的优雅度，使整句话更具感染力。', additional_kwargs={}, response_metadata={'model': 'qwen3:8b', 'created_at': '2025-12-09T09:07:18.43724Z', 'done': True, 'done_reason': 'stop', 'total_duration': 36826486958, 'load_duration': 83662500, 'prompt_eval_count': 58, 'prompt_eval_duration': 852940125, 'eval_count': 538, 'eval_duration': 35473802959, 'logprobs': None, 'model_name': 'qwen3:8b', 'model_provider': 'ollama'}, id='lc_run--019b025c-e739-7c21-8431-d9d02c8119c3-0', usage_metadata={'input_tokens': 58, 'output_tokens': 538, 'total_tokens': 596})]}
    # response['messages'][-1] 永远是 AI 的最新回复
    ai_message = response['messages'][-1]
   
    print("\n-------- 📝 翻译结果 --------")
    print(ai_message.content)
    print("----------------------")

    
    # 7. 查看 Token 消耗 (可选，用于调试)
    # 这里可以监控 Token 消耗，用于成本计算
    if hasattr(ai_message, 'usage_metadata'):
        usage = ai_message.usage_metadata
        print(f"\n📊 Token 消耗: Input {usage.get('input_tokens')} / Output {usage.get('output_tokens')}")

if __name__ == "__main__":
    test_translation()