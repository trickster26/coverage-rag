"""Azure AI Search backend: one index, hybrid query (BM25 + vector), jurisdiction
as a filter, optional semantic reranker.

Build the index:  python -m coverage_rag.azure_search
Requires AZURE_SEARCH_* and an Azure OpenAI embedding deployment.
"""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from .config import Settings, settings
from .ingest import Chunk, load_chunks
from .llm import embed_fn
from .search import Hit, term_coverage

SEMANTIC_CONFIG = "coverage-semantic"


def _index(name: str, dims: int) -> SearchIndex:
    return SearchIndex(
        name=name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="jurisdiction", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source", type=SearchFieldDataType.String),
            SimpleField(name="url", type=SearchFieldDataType.String),
            SimpleField(name="page", type=SearchFieldDataType.Int32),
            SearchableField(name="heading", type=SearchFieldDataType.String),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dims,
                vector_search_profile_name="hnsw-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
        ),
        semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
            name=SEMANTIC_CONFIG,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="heading"),
                content_fields=[SemanticField(field_name="text")],
            ),
        )]),
    )


class AzureSearchIndex:
    def __init__(self, s: Settings = settings):
        self.s = s
        self.embed = embed_fn(s)
        if self.embed is None:
            raise RuntimeError("Azure AI Search mode needs an embedding deployment (AZURE_OPENAI_EMBEDDING_DEPLOYMENT).")
        self.client = SearchClient(s.azure_search_endpoint, s.azure_search_index, AzureKeyCredential(s.azure_search_api_key))

    def search(self, query: str, jurisdiction: str | None = None, top_k: int = 5) -> list[Hit]:
        kwargs = {}
        if self.s.azure_search_semantic_config:
            kwargs = {"query_type": "semantic", "semantic_configuration_name": self.s.azure_search_semantic_config}
        results = self.client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=self.embed([query])[0],
                                            k_nearest_neighbors=self.s.candidates, fields="vector")],
            filter=f"jurisdiction eq '{jurisdiction}'" if jurisdiction in ("US", "IN") else None,
            top=top_k,
            **kwargs,
        )
        hits = []
        for r in results:
            chunk = Chunk(r["id"], r["jurisdiction"], r["source"], r["heading"], r["text"], r["url"], r.get("page"))
            score = r.get("@search.reranker_score") or r["@search.score"]
            hits.append(Hit(chunk, score, term_coverage(query, chunk.embed_text)))
        return hits


def build_index(s: Settings = settings) -> None:
    embed = embed_fn(s)
    if embed is None:
        raise RuntimeError("Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to build the Azure index.")
    chunks = load_chunks()
    dims = len(embed(["dimension probe"])[0])
    cred = AzureKeyCredential(s.azure_search_api_key)
    SearchIndexClient(s.azure_search_endpoint, cred).create_or_update_index(_index(s.azure_search_index, dims))
    client = SearchClient(s.azure_search_endpoint, s.azure_search_index, cred)
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        vectors = embed([c.embed_text for c in batch])
        client.upload_documents([{**c.__dict__, "vector": v} for c, v in zip(batch, vectors)])
    print(f"indexed {len(chunks)} chunks into {s.azure_search_index}")


if __name__ == "__main__":
    build_index()
