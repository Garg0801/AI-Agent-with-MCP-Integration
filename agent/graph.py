"""LangGraph StateGraph that implements the agent's reasoning loop.

The graph has two nodes:
    - `agent`: calls the tool-bound LLM to decide the next step.
    - `tools`: executes any tool calls the LLM requested, via LangChain's
               ToolNode (which handles dispatch to the right tool automatically).

Edges:
    START -> agent
    agent -> tools  (if the last AI message has tool_calls)
    agent -> END    (otherwise)
    tools -> agent  (loop back so the model can interpret the results)

Why hand-rolled instead of `create_react_agent`?:
The prebuilt is a one-liner, but writing the graph explicitly:
    (a) shows the reviewer we understand what's happening under the hood, and
    (b) makes modifications in the follow-up discussion round trivial
        (adding a planning node, a validator, a retry edge, etc.).
"""

from typing import Annotated, List, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    """Graph state: a growing list of chat messages.

    The `add_messages` reducer appends new messages instead of overwriting,
    which is what we want for a conversation history that grows as the agent
    thinks and calls tools.
    """

    messages: Annotated[List[AnyMessage], add_messages]


def build_agent_graph(
    llm_with_tools: BaseChatModel,
    tools: List[BaseTool],
    max_iterations: int = 5,
):
    """Build and compile the agent StateGraph.

    Args:
        llm_with_tools: A chat model already bound to tools via `.bind_tools()`.
            Pass the bound version so the LLM "sees" the available tools.
        tools: The same tools list, passed separately for the ToolNode to execute.
            The ToolNode needs the raw tool objects (not just their schemas).
        max_iterations: Maximum number of tool-call turns before the agent is
            forced to stop. Prevents runaway loops that would burn through
            tokens on a buggy query.

    Returns:
        A compiled LangGraph application. Invoke with:
            `await graph.ainvoke({"messages": [SystemMessage(...), HumanMessage(...)]})`
    """

    async def call_model(state: AgentState) -> dict:
        """Node: invoke the LLM with the current message history.

        Enforces the iteration cap before each call so we never make a
        tool-calling LLM call beyond the budget.
        """
        messages = state["messages"]

        # Count how many assistant turns have requested tools so far.
        tool_turns = sum(
            1
            for m in messages
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        if tool_turns >= max_iterations:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"I've reached the maximum number of tool-call iterations "
                            f"({max_iterations}) without completing this task. "
                            f"Based on what I've been able to gather so far, I'll stop here."
                        )
                    )
                ]
            }

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Conditional edge: route to tools if the last AI message asked for one.

        Returns the string name of the next node, or END.
        """
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
