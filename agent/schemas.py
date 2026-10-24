"""Pydantic models for configuration and telemetry.

Everything that crosses a module boundary - settings loaded from env,
MCP server configs, per-call telemetry records, and the final run report -
is a Pydantic model. This gives us validation, clear schemas, easy
JSON serialization, and IDE autocomplete across the whole project.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==============================================================
# Settings (loaded from .env + environment)
# ==============================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables and the .env file.

    pydantic-settings handles the loading, type coercion, and validation
    automatically. If HF_TOKEN is missing, startup fails fast with a clear
    error - much better than discovering it mid-run.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # HuggingFace / model configuration
    # hf_token: str = Field(..., description="HuggingFace API token for the Inference Providers router.")
    # model_name: str = Field(
    #     default="Qwen/Qwen2.5-72B-Instruct",
    #     description="Model ID on HuggingFace Hub.",
    # )
    # model_base_url: str = Field(
    #     default="https://router.huggingface.co/v1",
    #     description="OpenAI-compatible endpoint URL (HF router by default).",
    # )
    google_api_key: str = Field(..., description="Google AI (Gemini) API Key")

    model_name: str = Field(default="gemini-2.5-flash",
                            description="Gemini model name",)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    max_iterations: int = Field(
        default=5,
        gt=0,
        description="Maximum number of tool-call iterations before the agent is forced to stop.",
    )

    # MCP configuration
    mcp_sandbox_path: str = Field(
        default="./sandbox",
        description="Directory the filesystem MCP server is allowed to access.",
    )

    # LangSmith configuration (optional)
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: Optional[str] = Field(default=None)
    langsmith_project: str = Field(default="mcp-agent-assessment")


# ==============================================================
# MCP server configuration
# ==============================================================


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server over stdio transport.

    The langchain-mcp-adapters MultiServerMCPClient consumes a dict of these
    (serialized with .model_dump()) at construction time.
    """

    command: str = Field(..., description="Executable that launches the MCP server.")
    args: List[str] = Field(default_factory=list, description="CLI arguments for the server command.")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra env vars to pass to the subprocess.")
    transport: str = Field(default="stdio", description="MCP transport type (stdio|sse|streamable_http).")


# ==============================================================
# Telemetry records
# ==============================================================


class LLMCallRecord(BaseModel):
    """Captured metrics for one LLM invocation."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float


class ToolCallRecord(BaseModel):
    """Captured metrics for one tool invocation."""

    name: str
    server: str = "unknown"
    arguments: str = ""
    output: str = ""
    duration_seconds: float
    is_error: bool = False


class RunReport(BaseModel):
    """Aggregated telemetry for one full agent run (one user query -> final answer)."""

    query: str
    final_response: str
    llm_calls: List[LLMCallRecord] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_duration_seconds: float
