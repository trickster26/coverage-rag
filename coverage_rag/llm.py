"""Model clients through LangChain, so Azure OpenAI and OpenAI-compatible endpoints
are interchangeable behind one interface."""

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


def chat_model(s: Settings) -> BaseChatModel | None:
    if s.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_chat_deployment,
            temperature=0,
        )
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.openai_chat_model, api_key=s.openai_api_key,
                          base_url=s.openai_base_url or None, temperature=0)
    return None


def embed_fn(s: Settings):
    """Return texts -> vectors, or None when embeddings are not configured."""
    if not s.embeddings_enabled:
        return None
    if s.llm_provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        model = AzureOpenAIEmbeddings(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_embedding_deployment,
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        # Non-OpenAI servers expect raw strings, not tiktoken ids.
        model = OpenAIEmbeddings(model=s.openai_embedding_model, api_key=s.openai_api_key,
                                 base_url=s.openai_base_url or None,
                                 check_embedding_ctx_length=not s.openai_base_url)
    return model.embed_documents
