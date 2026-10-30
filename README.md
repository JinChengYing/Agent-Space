# Calculator Agent

一个基于 DeepSeek 工具调用的计算器 Agent，支持加、减、乘、除以及多步计算。

## 使用方法

1. 安装依赖：`uv sync`（或安装 `openai` 和 `python-dotenv`）。
2. 将 `calculate_agent/.env.example` 复制为 `calculate_agent/.env`，填入自己的 DeepSeek API key。不要提交 `.env`。
3. 运行：`uv run python calculate_agent/agent.py`。
4. 输入算式；输入 `exit` 或 `quit` 退出。

代码使用 DeepSeek 的 OpenAI 兼容接口。调用 API 可能产生费用。
