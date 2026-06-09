"""
ExamHelper 客户端智能体 — BYOA 实验主程序（DeepSeek 版）。

本程序完全保留原始 Gemini 版的架构与 Rich 渲染逻辑。
将 google-genai 驱动替换为 OpenAI 兼容驱动以连接 DeepSeek-V4-Flash。
连接 server.py 的 FastMCP 服务，醒目展示 Tool Call 原始报文。
"""
from __future__ import annotations
import os

# --- 代理配置 ---
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv
# --- 修改点：引入 OpenAI 替代 google-genai ---
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
SERVER_SCRIPT: Final[Path] = PROJECT_ROOT / "server.py"
# 使用 DeepSeek 最新 V4 模型
DEFAULT_MODEL: Final[str] = "deepseek-v4-flash" 
DEEPSEEK_BASE_URL: Final[str] = "https://api.deepseek.com"
MAX_RETRIES: Final[int] = 2

SYSTEM_INSTRUCTION: Final[str] = (
    "你是 ExamHelper 考试助手智能体，已通过 MCP 协议连接 ExamHelperServer。\n"
    "当用户询问考试大纲、复习重点时，请调用 analyze_local_syllabus 工具；"
    "当用户询问数据库或计算机专业概念（如 ACID、openGauss、JOIN 等）时，"
    "请调用 search_academic_concept 工具获取权威定义，避免凭空编造。\n"
    "项目根目录下有 database_syllabus.txt 可供分析。\n"
    "回答时请结合工具返回内容，条理清晰、专业准确。"
)

console: Console = Console(highlight=True)


# ---------------------------------------------------------------------------
# MCP 工具转换 & 结果提取
# ---------------------------------------------------------------------------


def mcp_tools_to_openai_tool(mcp_tools: list[Tool]) -> list[dict[str, Any]]:
    """
    将 MCP 工具列表转换为 DeepSeek/OpenAI 可识别的 Tool 定义。
    保持与原函数 mcp_tools_to_gemini_tool 相同的逻辑结构。
    """
    declarations = []
    for tool in mcp_tools:
        schema = tool.inputSchema or {
            "type": "object",
            "properties": {},
        }
        declarations.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            }
        })
    return declarations


def extract_tool_result_text(result: CallToolResult) -> str:
    """从 MCP CallToolResult 中提取可读文本。保持原样不动。"""
    text_parts: list[str] = []
    for block in result.content:
        if block.type == "text":
            text_parts.append(block.text)
        else:
            text_parts.append(str(block))
    if text_parts:
        return "\n".join(text_parts)
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, ensure_ascii=False, indent=2)
    return "(工具未返回任何内容)"


# ---------------------------------------------------------------------------
# 终端展示 (完全保留原代码的打印风格与报文拦截逻辑)
# ---------------------------------------------------------------------------


def build_tool_call_payload(
    tool_call: Any,
    *,
    model: str,
    turn_index: int,
) -> dict[str, Any]:
    """
    构造 Tool Call 原始报文，适配 DeepSeek。
    保持原代码 build_gemini_tool_call_payload 的 JSON 嵌套深度。
    """
    return {
        "report_meta": {
            "event": "deepseek_tool_call_intercept",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "turn_index": turn_index,
        },
        "deepseek_v4_raw": {
            "id": getattr(tool_call, 'id', 'unknown'),
            "method": getattr(tool_call.function, 'name', 'unknown'),
            "arguments": json.loads(getattr(tool_call.function, 'arguments', '{}')),
        },
    }


def print_tool_call_raw(
    tool_call: Any,
    *,
    model: str,
    turn_index: int,
) -> None:
    """
    醒目打印 [DeepSeek 发出的 Tool Call 原始报文]（JSON 高亮，保持金黄色面板）。
    """
    payload = build_tool_call_payload(
        tool_call,
        model=model,
        turn_index=turn_index,
    )
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)

    console.print()
    console.print(
        Rule(
            "[bold bright_yellow]★ DeepSeek 发出的 Tool Call 原始报文 ★[/bold bright_yellow]",
            characters="═",
            style="bright_yellow",
        )
    )
    console.print(
        Panel(
            Syntax(json_text, "json", theme="monokai", word_wrap=True),
            title="[bold bright_yellow on black] 📨 [DeepSeek 发出的 Tool Call 原始报文] [/bold bright_yellow on black]",
            subtitle="[dim]DeepSeek-V4-Flash · OpenAI 协议拦截[/dim]",
            border_style="bright_yellow",
            padding=(1, 2),
        )
    )
    console.print(Rule(style="bright_yellow"))


def print_tool_result_markdown(tool_name: str, result_text: str) -> None:
    """以 Markdown 面板展示 MCP 工具执行结果。保持原样不动。"""
    markdown_body = (
        f"## 工具执行结果\n\n"
        f"**工具名称**: `{tool_name}`  \n"
        f"**执行状态**: 成功\n\n"
        f"---\n\n"
        f"{result_text.strip()}\n"
    )
    console.print(
        Panel(
            Markdown(markdown_body),
            title=f"[bold green]🛠 MCP 工具响应 — {tool_name}[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def print_assistant_markdown(content: str) -> None:
    """以 Markdown 面板展示最终文本回复。保持原样不动。"""
    markdown_body = f"## 助手回复\n\n{content.strip()}\n"
    console.print(
        Panel(
            Markdown(markdown_body),
            title="[bold magenta]💬 DeepSeek 回复[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )


def print_network_error(message: str, *, detail: str | None = None) -> None:
    """打印网络/API 不稳定时的友好错误提示。保持原样不动。"""
    body = (
        f"[bold red]网络或 API 连接异常[/bold red]\n\n"
        f"{message}\n\n"
        "[dim]建议：\n"
        "  1. 检查网络连接与代理设置\n"
        "  2. 确认 DEEPSEEK_API_KEY 有效\n"
        "  3. 检查 api.deepseek.com 是否可达[/dim]"
    )
    if detail:
        body += f"\n\n[dim]详情: {detail}[/dim]"
    console.print(Panel(body, title="⚠ 连接提示", border_style="red"))


# ---------------------------------------------------------------------------
# DeepSeek + MCP 智能体客户端
# ---------------------------------------------------------------------------


class ExamAgentClient:
    """BYOA 实验客户端。完全保留原 ExamAgentClient 的结构。"""

    def __init__(self, *, api_key: str, model: str) -> None:
        self.model = model
        # 换成 OpenAI 客户端
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.ds_tools: list[dict] = []
        # OpenAI 格式的消息列表
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        self._turn_index = 0

    async def connect(self, server_script: Path = SERVER_SCRIPT) -> None:
        """保持原 connect 逻辑不动。"""
        if not server_script.is_file():
            raise FileNotFoundError(f"找不到 MCP 服务脚本: {server_script}")

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script)],
            env=None,
        )

        read_stream, write_stream = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

        tools_response = await self.session.list_tools()
        # 转换为适配 DeepSeek 的格式
        self.ds_tools = mcp_tools_to_openai_tool(tools_response.tools)
        tool_names = [tool.name for tool in tools_response.tools]

        console.print(
            Panel(
                f"[green]✓[/green] 已连接 [bold]ExamHelperServer[/bold]\n"
                f"服务脚本: [dim]{server_script}[/dim]\n"
                f"已注册工具 ({len(tool_names)}): "
                f"[cyan]{', '.join(tool_names)}[/cyan]",
                title="[bold]MCP 连接成功[/bold]",
                border_style="blue",
            )
        )

    async def _call_llm_with_retry(self) -> Any:
        """
        保留原代码 _call_gemini_with_retry 的重试包装逻辑。
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 对应原代码的 generate_content
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.ds_tools,
                    tool_choice="auto"
                )
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    console.print(f"[yellow]API 请求失败（第 {attempt} 次），正在重试…[/yellow]")
                    await asyncio.sleep(2 * attempt)
                    continue
                print_network_error("DeepSeek API 返回错误。", detail=str(exc))
                raise RuntimeError(str(exc)) from exc
        raise RuntimeError(f"调用失败: {last_error}")

    async def _execute_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """保持原 _execute_mcp_tool 逻辑不动。"""
        if self.session is None:
            raise RuntimeError("MCP 会话未初始化。")

        try:
            result: CallToolResult = await self.session.call_tool(
                tool_name,
                arguments=arguments,
            )
            return extract_tool_result_text(result)
        except Exception as exc:
            return f"工具执行失败: {exc}"

    async def process_user_message(self, user_text: str) -> None:
        """
        保持原 process_user_message 的 while True 状态机逻辑不动。
        """
        self._turn_index += 1
        self.messages.append({"role": "user", "content": user_text})

        while True:
            try:
                # 对应原代码调用 _call_gemini_with_retry
                response = await self._call_llm_with_retry()
            except RuntimeError:
                return

            response_msg = response.choices[0].message
            self.messages.append(response_msg)

            # 对应原代码 if response.function_calls:
            if response_msg.tool_calls:
                for call in response_msg.tool_calls:
                    # 1. 打印原始报文（对应原代码 print_gemini_tool_call_raw）
                    print_tool_call_raw(
                        call,
                        model=self.model,
                        turn_index=self._turn_index,
                    )

                    tool_name = call.function.name
                    tool_args = json.loads(call.function.arguments)

                    # 2. 执行工具
                    result_text = await self._execute_mcp_tool(tool_name, tool_args)
                    print_tool_result_markdown(tool_name, result_text)

                    # 3. 反馈结果给模型 (对应原 role="tool")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": tool_name,
                        "content": result_text,
                    })
                
                # 对应原代码 continue
                continue

            # 对应原代码 final_content = response.text
            final_content = response_msg.content or ""
            if final_content:
                print_assistant_markdown(final_content)
            break

    async def chat_loop(self) -> None:
        """保持原 chat_loop 样式不动。"""
        console.print(
            Panel(
                "[bold]ExamHelper BYOA 客户端（DeepSeek V4）已就绪[/bold]\n\n"
                f"模型: [cyan]{self.model}[/cyan]\n\n"
                "示例问题:\n"
                "  • 请分析 database_syllabus.txt 的考试重点\n"
                "  • ACID 特性分别是什么？\n"
                "  • openGauss 是什么数据库？\n\n"
                "[dim]输入 exit / quit / q / 退出 结束会话[/dim]",
                border_style="bright_blue",
            )
        )

        while True:
            try:
                console.print()
                user_input = console.input("[bold cyan]你 › [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]会话已中断。[/dim]")
                break

            if not user_input: continue
            if user_input.lower() in {"exit", "quit", "q", "退出"}:
                console.print("[dim]再见！[/dim]")
                break

            try:
                await self.process_user_message(user_input)
            except Exception as exc:
                console.print(Panel(f"[red]处理失败:[/red] {exc}", title="错误", border_style="red"))

    async def close(self) -> None:
        """保持原 close 逻辑。"""
        await self.exit_stack.aclose()


# ---------------------------------------------------------------------------
# 入口 (保持原 load_config 与 async_main 结构)
# ---------------------------------------------------------------------------


def load_config() -> tuple[str, str]:
    load_dotenv()
    # 改为获取 DeepSeek Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[bold red]错误:[/bold red] 请设置环境变量 [cyan]DEEPSEEK_API_KEY[/cyan]")
        raise SystemExit(1)

    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    return api_key, model


async def async_main() -> None:
    api_key, model = load_config()
    client = ExamAgentClient(api_key=api_key, model=model)

    try:
        await client.connect()
        await client.chat_loop()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(async_main())