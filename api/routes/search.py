# api/routes/search.py
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import AuthContext, get_auth_context, get_search_service
from api.schemas.search import SearchRequest, SearchResponse, SearchResultSchema
from src.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    results = search_service.search(
        query=payload.query,
        top_k=payload.top_k,
        use_reranking=payload.rerank,
        tenant_id=auth_context.tenant_id,
    )

    serialized = [SearchResultSchema.model_validate(item) for item in results]

    return SearchResponse(
        query=payload.query,
        results=serialized,
        total=len(serialized),
    )