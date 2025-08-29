import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()
MODEL = "deepseek-chat"

class MultiTurnReactAgent:
    def __init__(self):
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=MODEL,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # 初始化搜索工具
        self.search_tool = DuckDuckGoSearchRun()

        # 自定义计算工具
        self.calculator_tool = Tool(
            name="Calculator",
            func=self._calculate,
            description="用于执行数学计算，输入数学表达式"
        )

        # 定义工具列表
        self.tools = [self.search_tool, self.calculator_tool]

        # 初始化记忆
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # 从LangChain Hub获取ReAct提示
        self.prompt = hub.pull("hwchase17/react-chat")

        # 创建ReAct Agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)

        # 创建Agent执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def _calculate(self, expression):
        """简单的计算工具"""
        try:
            return str(eval(expression))
        except:
            return "计算错误，请检查表达式"

    def chat(self, user_input):
        """处理用户输入并返回响应"""
        try:
            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": self.memory.chat_memory.messages})
            return response["output"]
        except Exception as e:
            return f"处理请求时出错: {str(e)}"


# 使用示例
def main():
    # 初始化agent
    agent = MultiTurnReactAgent()

    print("React Agent 多轮对话示例")
    print("输入 'quit' 退出对话")
    print("-" * 50)

    # 模拟多轮对话
    conversations = [
        "帮我研究一下特斯拉最近的股价表现",
        "那特斯拉的市值现在是多少？",
        "帮我计算一下如果我现在投资10000美元，能买多少股？",
        "好的，那再帮我看看苹果公司的股价对比"
    ]

    for i, question in enumerate(conversations, 1):
        print(f"\n用户 [{i}]: {question}")
        response = agent.chat(question)
        print(f"Agent [{i}]: {response}")
        print("-" * 50)


if __name__ == "__main__":
    main()
