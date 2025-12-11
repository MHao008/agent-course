import os
import time
from dotenv import load_dotenv

# 导入 LangChain 核心组件
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 导入 LCEL 神器
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, chain

# 加载环境变量
load_dotenv()

def get_model():
    """初始化模型"""
    return init_chat_model(
        os.getenv("MODEL_NAME"),
        temperature=0.7,
        api_key=os.getenv("API_KEY"),
        model_provider=os.getenv("MODEL_PROVIDER", "ollama"),
        base_url=os.getenv("BASE_URL"),
    )

def test_magic_1_linear():
    """魔法一：基础线性链 (Linear Chain)"""
    print("\n--- 🪄 魔法一：基础线性链 ---")
    
    model = get_model()
    prompt = ChatPromptTemplate.from_template("请为一家生产 {product} 的公司起一个好听的中文名字。只返回名字，不要其他废话。")
    parser = StrOutputParser()

    # 🔗 组装：Prompt -> Model -> Parser
    # 这就是最经典的 LCEL 范式
    chain = prompt | model | parser

    # 调用
    result = chain.invoke({"product": "高性能显卡"})
    print(f"产品: 高性能显卡 -> 公司名: {result}")
    
    # ✨ 隐藏技巧：打印链条结构
    print("\n[Debug] 链条结构图:")
    chain.get_graph().print_ascii()
    
    return chain

def test_magic_2_custom_func():
    """魔法二：插入自定义函数 (@chain)"""
    print("\n--- 🪄 魔法二：插入自定义函数 ---")
    
    model = get_model()
    prompt = ChatPromptTemplate.from_template("翻译成英文: {text}")
    
    # 定义一个自定义的 Runnable 函数
    # @chain 装饰器会自动返回一个 Runnable 对象。等同于 RunnableLambda(add_prefix) 
    @chain
    def add_prefix(text):
        return f"✨ 结果: {text.strip()} ✨"

    # 组装：Prompt -> Model -> StrOutputParser -> 自定义函数
    pipeline = prompt | model | StrOutputParser() | add_prefix
    
    result = pipeline.invoke({"text": "你好，LangChain"})
    print(result)

def test_magic_3_passthrough():
    """魔法三：透传 (Passthrough) —— 解决上下文丢失问题"""
    print("\n--- 🪄 魔法三：透传 (Passthrough) ---")
    
    model = get_model()
    parser = StrOutputParser()
    
    # 步骤1：生成名字
    name_prompt = ChatPromptTemplate.from_template("请为一家生产 {product} 的公司起一个好听的中文名字。只返回名字。")
    generate_name_chain = name_prompt | model | parser

    # 步骤2：写 Slogan
    # 注意：这个 Prompt 需要 {company_name} (上一步生成的) 和 {product} (最开始输入的)
    slogan_prompt = ChatPromptTemplate.from_template(
        "公司名是：{company_name}，产品是：{product}。请写一句朗朗上口的 Slogan（口号）。"
    )

    # 🔗 组装复杂链
    # 使用字典结构，RunnablePassthrough() 代表"原始输入"
    full_chain = (
        {"product": RunnablePassthrough(), "company_name": generate_name_chain} 
        | slogan_prompt 
        | model 
        | parser
    )

    result = full_chain.invoke("量子计算机")
    print(f"结果: {result}")

def test_magic_4_parallel():
    """魔法四：并行处理 (Parallel) —— 效率倍增"""
    print("\n--- 🪄 魔法四：并行处理 (Parallel) ---")
    
    model = get_model()
    parser = StrOutputParser()

    # 定义两个并行的链
    pros_chain = ChatPromptTemplate.from_template("简短列出 {product} 的一个核心优点") | model | parser
    cons_chain = ChatPromptTemplate.from_template("简短列出 {product} 的一个核心缺点") | model | parser

    # 🔗 组装并行链
    # 就像电路并联一样，两路同时跑
    map_chain = RunnableParallel(
        pros=pros_chain,
        cons=cons_chain
    )

    start_time = time.time()
    print("⏳ 开始并行思考...")
    
    # 调用
    result = map_chain.invoke({"product": "纯电动汽车"})
    
    end_time = time.time()
    print(f"✅ 完成! 耗时: {end_time - start_time:.2f}秒")
    print(f"优点: {result['pros']}")
    print(f"缺点: {result['cons']}")


def test_stream():
    """魔法五：流式输出 (Streaming)"""
    print("\n--- 🪄 魔法五：流式输出 (Streaming) ---")
    
    model = get_model()
    prompt = ChatPromptTemplate.from_template("公司名是：{company_name}，产品是：{product}。请写一句朗朗上口的 Slogan（口号）。")
    parser = StrOutputParser()

    # 🔗 组装：Prompt -> Model -> Parser
    # 这就是最经典的 LCEL 范式
    chain = prompt | model | parser

    # 调用
    for chunk in chain.stream({"company_name": "芯擎", "product": "高性能显卡"}):
        if chunk:
            # chunk 是实时吐出的字符
            print(chunk, end="|", flush=True)
    
    return chain


if __name__ == "__main__":
    # 按需运行测试
    # test_magic_1_linear()
    # test_magic_2_custom_func()
    # test_magic_3_passthrough()
    # test_magic_4_parallel()
    test_stream()