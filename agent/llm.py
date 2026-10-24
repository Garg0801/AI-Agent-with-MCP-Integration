"""LLM factory.

Isolated in its own module so swapping the model (or the provider entirely)
is a one-file change -- a common ask in the follow-up discussion round.
"""

# from langchain_openai import ChatOpenAI

# from agent.schemas import Settings

from langchain_google_genai import ChatGoogleGenerativeAI
from agent.schemas import Settings

def build_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    """Build a ChatGoogleGenerativeAI client for Gemini.

    Uses langchain-google-genai (the official LangChain integration) rather
    than ChatOpenAI pointed at Gemini's OpenAI compatibility layer, because
    the OpenAI compat endpoint has known function-calling issues specifically
    with gemini-2.5-flash-preview. The native integration goes through Google's
    full API and handles function calling reliably.

    Args:
        settings: Validated Settings instance.

    Returns:
        A configured ChatGoogleGenerativeAI ready for .bind_tools() and .ainvoke().
    """
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
        timeout=60,
        max_retries=2,
    )


# def build_llm(settings: Settings) -> ChatOpenAI:
#     """Build a ChatOpenAI client pointed at the HuggingFace Inference Providers router.

#     Why ChatOpenAI and not ChatHuggingFace?
#     HuggingFace exposes an OpenAI-compatible endpoint at `router.huggingface.co/v1`,
#     which lets us use the battle-tested `langchain-openai` client with open-source
#     models like Qwen 2.5 or Llama 3.3. For tool calling specifically this path is
#     more reliable than ChatHuggingFace because the OpenAI protocol is the native
#     tool-call format for these models' serving backends (Sambanova, Together,
#     Fireworks, etc.) -- one fewer translation layer.

#     Args:
#         settings: Validated Settings instance (from agent.schemas).

#     Returns:
#         A configured ChatOpenAI ready for `.bind_tools()` and `.ainvoke()`.
#     """
#     return ChatOpenAI(
#         model=settings.model_name,
#         api_key=settings.hf_token,
#         base_url=settings.model_base_url,
#         temperature=settings.temperature,
#         max_tokens=settings.max_tokens,
#         # Timeouts help with occasional slow routes through HF providers.
#         timeout=60,
#         max_retries=2,
#     )
