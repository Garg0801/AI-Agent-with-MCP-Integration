"""MCP server connection management with explicit startup diagnostics."""

from typing import Dict, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.schemas import MCPServerConfig


class MCPManager:
    """Connect to configured stdio MCP servers and collect their tools."""

    def __init__(self, server_configs: Dict[str, MCPServerConfig]):
        self.server_configs = server_configs
        self._client: MultiServerMCPClient | None = None
        self._tools: List[BaseTool] = []
        self._tool_to_server: Dict[str, str] = {}

    async def connect(self) -> List[BaseTool]:
        """Load each server's tools, preserving server attribution."""
        config_dict = {
            name: cfg.model_dump(exclude_none=True)
            for name, cfg in self.server_configs.items()
        }
        self._client = MultiServerMCPClient(config_dict)

        all_tools: List[BaseTool] = []
        self._tool_to_server.clear()

        for server_name in self.server_configs:
            print(f"\nConnecting to MCP server: {server_name}")
            try:
                server_tools = await self._client.get_tools(server_name=server_name)
            except Exception as exc:
                print(f"[FAILED] MCP server '{server_name}': {type(exc).__name__}: {exc}")
                raise RuntimeError(
                    f"Unable to initialize MCP server '{server_name}'. "
                    "Check its command, arguments, package startup, and stderr."
                ) from exc

            tool_names = [tool.name for tool in server_tools]
            print(f"[SUCCESS] {server_name}: {len(server_tools)} tools loaded")
            for tool in server_tools:
                print(f"  - {tool.name}")
                self._tool_to_server[tool.name] = server_name
            all_tools.extend(server_tools)

            # A stdio server that initializes but advertises no tools is usually
            # misconfigured or not the server/package expected by this project.
            if not server_tools:
                print(
                    f"[WARNING] MCP server '{server_name}' initialized but "
                    "advertised zero tools."
                )

        self._tools = all_tools
        if not self._tools:
            raise RuntimeError(
                "MCP servers connected but returned zero tools. "
                "Check the per-server startup messages above."
            )
        return self._tools

    @property
    def tools(self) -> List[BaseTool]:
        return self._tools

    @property
    def tool_to_server(self) -> Dict[str, str]:
        return self._tool_to_server

    def get_server_for_tool(self, tool_name: str) -> str:
        return self._tool_to_server.get(tool_name, "unknown")

    def tool_summary(self) -> Dict[str, List[str]]:
        summary: Dict[str, List[str]] = {name: [] for name in self.server_configs}
        for tool in self._tools:
            server = self.get_server_for_tool(tool.name)
            summary.setdefault(server, []).append(tool.name)
        return summary

    async def close(self) -> None:
        """Release the MCP client if the installed adapter exposes cleanup."""
        if self._client is None:
            return
        for method_name in ("aclose", "close", "shutdown"):
            method = getattr(self._client, method_name, None)
            if method is None:
                continue
            try:
                result = method()
                if hasattr(result, "__await__"):
                    await result
                return
            except Exception:
                continue
