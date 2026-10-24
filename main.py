"""Main entrypoint - runs the three required demo scenarios.

This script:
    1. Loads settings from .env (via pydantic-settings).
    2. Prepares a sandbox directory for the filesystem MCP server.
    3. Connects to two MCP servers (filesystem + time) and loads their tools.
    4. Builds the configured Gemini model through the Google AI API.
    5. Compiles a LangGraph agent that lets the LLM decide which tools to use.
    6. Runs three scenarios, captures telemetry via a callback handler,
       and prints a per-scenario report plus a JSON dump of all runs.

The three scenarios match the assessment's demonstration requirements:
    - Scenario 1: agent answers without any tool.
    - Scenario 2: agent determines a tool is needed and calls it.
    - Scenario 3: agent chains tools across two different MCP servers.

Run with: `python main.py`
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from agent.graph import build_agent_graph
from agent.llm import build_llm
from agent.mcp_manager import MCPManager
from agent.prompts import SYSTEM_PROMPT
from agent.schemas import MCPServerConfig, Settings
from agent.telemetry import TelemetryCallbackHandler


# ==============================================================
# Scenarios (match the three required demonstrations in the brief)
# ==============================================================

SCENARIOS: List[Dict[str, str]] = [
    {
        "id": "1",
        "name": "No tool needed",
        "description": "The agent should answer directly from its own knowledge, with zero tool calls.",
        "query": "What is the capital of France? Answer in one short sentence.",
    },
    {
        "id": "2",
        "name": "Clear tool call",
        "description": "The agent should determine that an MCP tool is required and call it.",
        "query": "List all files in the sandbox root directory.",
    },
    {
        "id": "3",
        "name": "Multi-tool chaining across MCP servers",
        "description": "The agent should pick the right tool from two different MCP servers and chain them.",
        "query": (
            "Get the current time in UTC, then write a file called 'timestamp.txt' "
            "in the sandbox containing exactly that timestamp. Confirm when done."
        ),
    },
]

# ==============================================================
# Setup helpers
# ==============================================================


def configure_langsmith(settings: Settings, console: Console) -> None:
    """Export LangSmith env vars so LangChain picks them up automatically.

    LangSmith reads these at import/trace time, so we set them before any
    traced code runs. If tracing is disabled or no API key is set, we skip
    silently - the agent still works, just without the extra observability.
    """
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        console.print("[dim]LangSmith tracing disabled.[/dim]")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    # LangChain reads both LANGSMITH_* (new) and LANGCHAIN_* (old) prefixes;
    # set both to be safe across versions.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    console.print(
        f"[green]✔[/green] LangSmith tracing enabled -> project '{settings.langsmith_project}'"
    )


def ensure_sandbox(path: str, console: Console) -> None:
    """Create the sandbox directory with a couple of sample files to list/read.

    The filesystem MCP server will be restricted to this directory, so having
    some sample content lets scenarios 2 and 3 do something interesting.
    """
    sandbox = Path(path).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "sample.txt").write_text(
        "This is a sample file in the sandbox.\nLine two for good measure.\n"
    )
    (sandbox / "notes.md").write_text(
        "# Notes\n\nSome markdown notes used by the demo scenarios.\n"
    )
    console.print(f"[green]✔[/green] Sandbox ready at: {sandbox}")


def build_mcp_configs(settings: Settings) -> Dict[str, MCPServerConfig]:
    """Build Windows-compatible stdio configurations for both MCP servers."""
    import sys

    is_windows = sys.platform == "win32"
    sandbox_abs = str(Path(settings.mcp_sandbox_path).resolve())

    if is_windows:
        # Launch the npm .cmd shim through cmd.exe so Windows subprocess
        # creation works reliably with stdio-based MCP clients.
        filesystem_command = "cmd.exe"
        filesystem_args = [
            "/d",
            "/c",
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            sandbox_abs,
        ]
        time_command = "uvx.exe"
    else:
        filesystem_command = "npx"
        filesystem_args = [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            sandbox_abs,
        ]
        time_command = "uvx"

    return {
        "filesystem": MCPServerConfig(
            command=filesystem_command,
            args=filesystem_args,
            transport="stdio",
        ),
        "time": MCPServerConfig(
            command=time_command,
            args=["mcp-server-time", "--local-timezone", "UTC"],
            transport="stdio",
        ),
    }

# ==============================================================
# Scenario runner
# ==============================================================


async def run_scenario(
    graph: Any,
    scenario: Dict[str, str],
    telemetry: TelemetryCallbackHandler,
    tool_to_server: Dict[str, str],
    console: Console,
) -> Dict[str, Any]:
    """Run one scenario end-to-end and return the structured report dict."""
    console.print()
    console.print(Rule(f"[bold]Scenario {scenario['id']}: {scenario['name']}[/bold]"))
    console.print(f"[italic dim]{scenario['description']}[/italic dim]")
    console.print(Panel(scenario["query"], title="[bold cyan]User Query[/bold cyan]"))

    telemetry.reset()
    start = time.time()

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=scenario["query"]),
        ]
    }

    try:
        result = await graph.ainvoke(
            initial_state,
            config={"callbacks": [telemetry]},
        )
        final_message = result["messages"][-1]
        final_text = getattr(final_message, "content", str(final_message))
    except Exception as e:
        final_text = f"[ERROR] Agent execution failed: {e}"
        console.print(f"[bold red]Scenario failed:[/bold red] {e}")

    duration = time.time() - start
    if isinstance(final_text, str):
        final_response = final_text

    elif isinstance(final_text, list):
        final_response = "\n".join(
            block["text"]
            for block in final_text
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()

    else:
        final_response = str(final_text)

    report = telemetry.build_report(
        query=scenario["query"],
        final_response=final_response,
        total_duration=duration,
        tool_to_server=tool_to_server,
    )

    telemetry.print_report(report, console)
    return {"scenario": scenario, "report": report.model_dump()}

# ==============================================================
# Main
# ==============================================================


async def main() -> None:
    """Set up MCP + LLM + graph, run all scenarios, and save reports."""
    load_dotenv()
    settings = Settings()  # Validates the configured Google API key.
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]AI Agent with MCP integration[/bold cyan]\n"
        "[dim]Gemini • LangGraph • MCP • LangSmith[/dim]",
        border_style="cyan",
    ))

    # --- Setup ---
    configure_langsmith(settings, console)
    ensure_sandbox(settings.mcp_sandbox_path, console)

    # --- MCP servers ---
    mcp_configs = build_mcp_configs(settings)
    console.print("Connecting to MCP servers...")
    manager = MCPManager(mcp_configs)

    try:
        tools = await manager.connect()
        console.print(
            f"[green]✔[/green] Loaded [bold]{len(tools)}[/bold] tools from "
            f"[bold]{len(mcp_configs)}[/bold] MCP server(s):"
        )
        for server, tool_names in manager.tool_summary().items():
            pretty = ", ".join(tool_names) if tool_names else "(none)"
            console.print(f"    [cyan]{server}[/cyan] -> {pretty}")

        # --- LLM + graph ---
        llm = build_llm(settings)
        llm_with_tools = llm.bind_tools(tools)
        console.print(
            f"[green]✔[/green] LLM ready: [bold]{settings.model_name}[/bold] "
            "(via Google Gemini API)"
        )

        graph = build_agent_graph(
            llm_with_tools, tools, max_iterations=settings.max_iterations
        )
        console.print("[green]✔[/green] Agent graph compiled.")

        # --- Scenarios ---
        telemetry = TelemetryCallbackHandler(default_model_name=settings.model_name)
        all_reports: List[Dict[str, Any]] = []
        for scenario in SCENARIOS:
            report = await run_scenario(
                graph, scenario, telemetry, manager.tool_to_server, console
            )
            all_reports.append(report)

        # --- Save consolidated reports ---
        output_path = Path("run_reports.json")
        output_path.write_text(json.dumps(all_reports, indent=2, default=str))
        console.print()
        console.print(
            f"[green]✔[/green] Full reports saved to [bold]{output_path.resolve()}[/bold]"
        )

    finally:
        # Always clean up subprocesses, even on error.
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
