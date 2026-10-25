"""LLM factory.

Isolated in its own module so swapping the model (or the provider entirely)
is a one-file change.
"""


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

