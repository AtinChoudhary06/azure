"""
Run this once to create (or update) the Azure AI Search index.
Usage:  python scripts/create_index.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

from config import settings

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=settings.EMBEDDING_DIM,
        vector_search_profile_name="my-vector-profile",
    ),
    SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
    profiles=[
        VectorSearchProfile(
            name="my-vector-profile",
            algorithm_configuration_name="my-hnsw-config",
        )
    ],
)

index = SearchIndex(name=settings.SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)

client = SearchIndexClient(endpoint=settings.SEARCH_ENDPOINT, credential=AzureKeyCredential(settings.SEARCH_KEY))
result = client.create_or_update_index(index)

print(f"Index '{result.name}' created successfully with {len(result.fields)} fields.")