from  openai import OpenAI
from dotenv import load_dotenv
import os
import json
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"))  # 读取与脚本同目录的 .env
#初始化客户端
client=OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
) #创建客户端

#定义python工具
def calculator(a,b,operation):
    if operation == "+":
        return a+b
    elif operation == "-":
        return a-b
    elif operation == "*":
        return a*b
    elif operation == "/":
        if b != 0:
            return a/b
        else:
            return "错误：除数不能为0"
    return "不支持的运算"
#发送请求
#描述我们的工具（相当于写接口）
tools=[
    {
        "type":"function",
        "function":{
            "name":"calculator",
            "description":"执行两个数字的加减乘除运算",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{"type":"number"},
                    "b":{"type":"number"},
                "operation":{
                    "type":"string",
                    "enum":["+","-","*","/"]
                             }
                }
            },
            "required":["a","b","operation"]
        }
    }
]
#agent loop

def run_agent(query):
    """
    运行一个基于 DeepSeek 的简单 Agent。

    参数:
        query (str): 用户输入的任务。
                     例如："计算 (125 + 75) * 3"

    返回:
        str: Agent 最终生成的回答。

    核心流程:
        1. 初始化对话历史 messages
        2. 将 messages 发送给 DeepSeek
        3. DeepSeek 决定直接回答还是调用工具
        4. 如果调用工具，由 Python 执行工具
        5. 将工具结果追加到 messages
        6. 再次调用 DeepSeek，直到任务完成
    """

    # ==========================================
    # Step 1: 初始化 Agent 的对话历史
    # ==========================================

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个数学Agent。"
                "遇到数学计算必须使用calculator工具。"
                "可以通过多次调用工具完成复杂计算。"
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]

    # messages 是一个列表，用于存储当前任务的所有消息。
    #
    # 每条消息一般包含:
    #   role: 消息由谁产生
    #   content: 消息内容
    #
    # 常见 role:
    #   system    -> 系统指令
    #   user      -> 用户输入
    #   assistant -> DeepSeek 输出
    #   tool      -> 工具执行结果
    #
    # 需要注意:
    # messages 不只是普通聊天记录。
    # 它还会保存工具调用请求和工具执行结果。
    #
    # 模型每轮都能通过这些历史消息获得任务上下文。
    #
    # 这里 messages 是局部变量，因此每次调用
    # run_agent() 都会重新建立对话历史。


    # ==========================================
    # Step 2: 开始 Agent 的多轮执行循环
    # ==========================================

    for step in range(10):

        # 最多允许 Agent 执行 10 轮模型决策。
        #
        # 每轮可能发生:
        #
        #   情况 A: 模型给出最终答案 -> 结束循环
        #
        #   情况 B: 模型请求调用工具 -> 执行工具
        #           -> 返回结果 -> 进入下一轮
        #
        # 设置最大轮数，是为了防止 Agent
        # 反复调用工具而无法结束。

        print(f"\n===== Step {step + 1} =====")


        # ======================================
        # Step 3: 调用 DeepSeek，让模型做决策
        # ======================================

        response = client.chat.completions.create(

            # 指定使用哪个 DeepSeek 模型。
            model="deepseek-flash",

            # 将完整对话历史发送给模型。
            #
            # 包括:
            #   系统指令
            #   用户任务
            #   模型过去的输出
            #   工具调用结果
            messages=messages,

            # 向模型提供可以使用的工具定义。
            #
            # 注意:
            # tools 仅描述工具的名称、功能、参数。
            # 模型不能直接执行本地 Python 函数。
            tools=tools,

            # auto 表示由模型自主决定:
            #   1. 是否调用工具
            #   2. 调用哪个工具
            #   3. 使用哪些参数
            tool_choice="auto"
        )

        # response 是 API 返回的完整响应对象。
        #
        # 其 choices[0].message 表示当前模型
        # 输出的 assistant 消息。
        #
        # 它可能包含:
        #
        #   content: 普通文本回答
        #
        #   tool_calls: 模型请求调用的工具列表
        #
        # 如果模型需要使用工具，
        # tool_calls 通常不是空的。


        # ======================================
        # Step 4: 提取本轮模型消息
        # ======================================

        message = response.choices[0].message

        # 将模型消息转换为字典，
        # 然后保存到对话历史中。
        #
        # 对于工具调用消息，这一步非常重要。
        #
        # 因为模型的工具调用请求也属于对话历史，
        # 后续必须将对应的工具执行结果
        # 与该请求一起提供给模型。

        messages.append(
            message.model_dump(exclude_none=True)
        )


        # ======================================
        # Step 5: 判断模型是否需要调用工具
        # ======================================

        if not message.tool_calls:

            # 如果 tool_calls 为空，
            # 表示这轮模型没有请求工具调用。
            #
            # 在我们的简化 Agent 设计中，
            # 将这种情况视为任务已经完成。
            #
            # 注意:
            # 在生产环境还应该检查 finish_reason、
            # 内容是否为空，以及其他异常情况。

            print("Agent:", message.content)

            # 返回最终答案，终止 run_agent()
            return message.content


        # ======================================
        # Step 6: 遍历模型请求的工具
        # ======================================

        # 一次模型响应可能包含多个 tool_call。
        #
        # 因此这里不能假设每轮只调用一个工具。
        #
        # 例如模型可能同时请求:
        #
        #   calculator(10, 20, "add")
        #   calculator(5, 6, "multiply")

        for tool_call in message.tool_calls:

            # 读取模型希望调用的函数名称。
            #
            # 例如:
            #   "calculator"

            name = tool_call.function.name

            try:
                # 模型返回的工具参数是 JSON 字符串。
                #
                # 假设:
                # tool_call.function.arguments =
                # '{"a":125,"b":75,"operation":"add"}'
                #
                # json.loads 将字符串转换为 Python 字典。

                args = json.loads(
                    tool_call.function.arguments
                )

                # 此时:
                #
                # args = {
                #     "a": 125,
                #     "b": 75,
                #     "operation": "add"
                # }

                print("Action:", name)
                print("Arguments:", args)


                # ==================================
                # Step 7: 执行 Python 工具
                # ==================================

                # 注意:
                # 这里才是真正执行 Python 函数的地方。
                #
                # 前面 DeepSeek 只是生成了工具调用请求。
                #
                # 现在由 Agent 程序根据函数名称，
                # 选择对应的 Python 函数执行。

                if name == "calculator":

                    # **args 表示字典参数展开。
                    #
                    # calculator(**args)
                    #
                    # 等价于:
                    #
                    # calculator(
                    #     a=125,
                    #     b=75,
                    #     operation="add"
                    # )

                    result = calculator(**args)

                else:
                    # 如果模型请求了未注册的工具，
                    # 不执行该工具，返回错误。
                    result = "未知工具"

            except (ValueError, TypeError, KeyError) as e:

                # 捕获常见的工具参数解析或执行错误。
                #
                # 例如:
                #   JSON 格式错误
                #   参数缺失
                #   参数类型不正确
                #
                # 错误信息也会作为工具结果返回模型，
                # 让模型有机会调整下一步行动。

                result = f"工具执行错误: {e}"

            print("Observation:", result)


            # ==================================
            # Step 8: 将工具执行结果反馈给模型
            # ==================================

            messages.append({
                "role": "tool",

                # tool_call_id 用于关联:
                #
                # 模型提出的工具调用请求
                # 和
                # Python 返回的工具执行结果。
                #
                # 一次响应存在多个工具调用时，
                # 每个工具调用有自己的 id。

                "tool_call_id": tool_call.id,

                # 工具结果必须以消息形式提供。
                #
                # 这里将 Python 返回值转换为字符串。
                #
                # 例如:
                # calculator(...) 返回整数 200
                #
                # 则消息中的 content 为 "200"。

                "content": str(result)
            })

            # 至此完成了一次:
            #
            # Action -> Tool Execution -> Observation
            #
            # 这里不会立即对每个工具单独请求模型。
            # 而是先执行本轮全部工具调用，
            # 再进入外层循环，统一请求下一轮决策。


    # ==========================================
    # Step 9: 处理达到最大轮数的情况
    # ==========================================

    # 如果经过 10 轮决策，
    # 模型始终没有生成最终答案，
    # 则停止执行。
    #
    # 这是一个简单的安全限制，
    # 防止无限循环和过多 API 消耗。

    return "达到最大执行轮数"

if __name__ == "__main__":
    while True:
            query = input("\nUser: ")
            if query.lower() in ["exit", "quit"]:
                break

            run_agent(query)#获取模型输出的文本
