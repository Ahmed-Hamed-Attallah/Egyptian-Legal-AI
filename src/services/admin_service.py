from __future__ import annotations
from typing import Any, Dict, Optional
from src.config.settings import settings

class AdminService:
    def __init__(
        self,
        chat_service: Optional[Any] = None,
        search_service: Optional[Any] = None,
        document_service: Optional[Any] = None,
        provider_router: Optional[Any] = None,
        database: Optional[Any] = None,
    ) -> None:
        self.chat_service = chat_service
        self.search_service = search_service
        self.document_service = document_service
        self.provider_router = provider_router
        self.database = database

    def health(self) -> Dict[str, Any]:
        components: Dict[str, str] = {}
        components["chat"] = "ok" if self.chat_service else "unavailable"
        components["search"] = "ok" if self.search_service else "unavailable"
        components["documents"] = "ok" if self.document_service else "unavailable"
        
        components["postgres"] = "unknown"
        if self.database:
            try:
                components["postgres"] = "ok" if self.database.health_check() else "unreachable"
            except Exception:
                components["postgres"] = "error"
                
        components["qdrant"] = "ok" if self.search_service else "unavailable"
        
        overall = "ok" if all(value == "ok" for value in components.values()) else "degraded"
        return {"status": overall, "components": components}

    def status(self) -> Dict[str, Any]:
        llm_status = {
            "primary_provider": getattr(settings, "LLM_PRIMARY_PROVIDER", "unknown"),
            "fallback_order": getattr(settings, "LLM_FALLBACK_ORDER", "unknown"),
            "available": [],
            "unavailable": [],
            "missing_credentials": [],
        }
        
        if self.provider_router and hasattr(self.provider_router, "get_provider_statuses"):
            statuses = self.provider_router.get_provider_statuses()
            for name, info in statuses.items():
                if info.get("initialized"):
                    llm_status["available"].append(name)
                else:
                    llm_status["unavailable"].append(name)
        else:
            for p in ["groq", "openai", "gemini"]:
                key = f"{p.upper()}_API_KEY"
                if not getattr(settings, key, None):
                    llm_status["missing_credentials"].append(p)
                    llm_status["unavailable"].append(p)
                else:
                    llm_status["available"].append(p)
                    
        qdrant_status = "ok" if (self.search_service and getattr(self.search_service, "hybrid_retriever", None)) else "degraded_or_uninitialized"
            
        postgres_status = "unknown"
        if self.database:
            try:
                postgres_status = "ok" if self.database.health_check() else "unreachable"
            except Exception:
                postgres_status = "error"
                
        return {
            "llm": llm_status,
            "rag": {
                "initialized": self.chat_service is not None and self.search_service is not None,
                "chat_service": "ok" if self.chat_service else "unavailable",
                "search_service": "ok" if self.search_service else "unavailable",
                "document_service": "ok" if self.document_service else "unavailable",
            },
            "qdrant": qdrant_status,
            "postgres": postgres_status,
            "settings": {
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "embedding_device": settings.EMBEDDING_DEVICE,
                "qdrant_collection": settings.QDRANT_COLLECTION_NAME,
                "vector_dimension": settings.VECTOR_DIMENSION,
                "reranking_model": settings.RERANKING_MODEL_NAME,
                "auth_enabled": settings.AUTH_ENABLED,
                "cache_enabled": settings.CACHE_ENABLED,
                "document_storage_dir": str(settings.DOCUMENT_STORAGE_DIR),
                "max_upload_size_bytes": settings.MAX_UPLOAD_SIZE_BYTES,
                "allowed_file_types": settings.ALLOWED_FILE_TYPES,
                "worker_enabled": settings.WORKER_ENABLED,
                "worker_embedded": settings.WORKER_EMBEDDED,
                "metrics_enabled": settings.METRICS_ENABLED,
            }
        }