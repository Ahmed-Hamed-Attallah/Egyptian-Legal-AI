# src/vector_store/qdrant_store.py
"""
Concrete Qdrant implementation of the Vector Store.
"""
import uuid
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, Filter, FieldCondition, MatchValue

from src.vector_store.base import BaseVectorStore
from src.vector_store.collection_manager import CollectionManager
from src.domain.chunks import Chunk, ChunkType
from src.domain.queries import RetrievedChunk
from src.domain.exceptions import VectorDatabaseError
from src.retrieval.metadata_filters import MetadataFilter
from src.config.settings import settings

logger = logging.getLogger(__name__)

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant-based vector storage. Handles connection, collection initialization,
    and point operations while preserving legal metadata.
    """
    def __init__(self, client: Optional[QdrantClient] = None, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.client = client or self._create_client()
        self.collection_manager = CollectionManager(self.client)
        self._ensure_collection()


    def _ensure_collection(self):
        self.collection_manager.ensure_collection(
            collection_name=self.collection_name,
            vector_size=settings.VECTOR_DIMENSION,
            distance=self._get_distance_metric()
        )

    def _get_distance_metric(self) -> Distance:
        metric_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT
        }
        return metric_map.get(settings.DISTANCE_METRIC.lower(), Distance.COSINE)

    def _generate_point_id(self, chunk: Chunk) -> str:
        """Generates a deterministic UUID for a chunk based on its document ID and index."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.document_id}_{chunk.chunk_index}"))

    def _chunk_to_point(self, chunk: Chunk, embedding: List[float]) -> PointStruct:
        metadata = getattr(chunk, "metadata", None) or {}

        point_id = metadata.get("vector_id") or metadata.get("point_id")
        if not point_id:
            point_id = self._generate_point_id(chunk)
        else:
            point_id = str(point_id)

        document_id = getattr(chunk, "document_id", None) or metadata.get("document_id")
        chunk_index = getattr(chunk, "chunk_index", None)
        if chunk_index is None:
            chunk_index = metadata.get("chunk_index", 0)

        chunk_type = getattr(chunk, "chunk_type", None)
        if chunk_type is not None:
            chunk_type = str(chunk_type)
        else:
            chunk_type = metadata.get("chunk_type")

        payload = {
            "document_id": str(document_id) if document_id is not None else None,
            "chunk_index": int(chunk_index or 0),
            "content": chunk.content,
            "chunk_type": chunk_type,
            "article_number": getattr(chunk, "article_number", None)
            or metadata.get("article_number"),
            "section_title": getattr(chunk, "section_title", None)
            or metadata.get("section_title"),
        }

        # Add important legal metadata for retrieval/citation/filtering.
        for key in (
            "tenant_id",
            "chunk_id",
            "law_name",
            "law_number",
            "law_year",
            "page_number",
            "source_path",
            "storage_key",
            "vector_id",
        ):
            if key in metadata and key not in payload:
                payload[key] = metadata[key]

        # Flatten remaining custom metadata.
        reserved_keys = {
            "document_id",
            "chunk_index",
            "content",
            "chunk_type",
            "article_number",
            "section_title",
            "tenant_id",
            "chunk_id",
            "law_name",
            "law_number",
            "law_year",
            "page_number",
            "source_path",
            "storage_key",
            "vector_id",
        }

        for k, v in metadata.items():
            if k not in reserved_keys:
                payload[k] = v

        # Remove None values to keep payload clean.
        payload = {k: v for k, v in payload.items() if v is not None}

        return PointStruct(id=point_id, vector=embedding, payload=payload)

    def insert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        if len(chunks) != len(embeddings):
            raise VectorDatabaseError("Number of chunks and embeddings must match")

        points = [self._chunk_to_point(c, e) for c, e in zip(chunks, embeddings)]
        point_ids = [str(point.id) for point in points]

        try:
            # Using upsert to handle any potential duplicates gracefully.
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                "Inserted/Updated %d points into '%s'",
                len(points),
                self.collection_name,
            )
            return point_ids
        except Exception as exc:
            raise VectorDatabaseError(f"Failed to insert points: {exc}") from exc

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[RetrievedChunk]:
        """
        Search Qdrant using the available client API.

        Supports:
        - Older qdrant-client: client.search(...)
        - Newer qdrant-client: client.query_points(...)
        """
        qdrant_filter = None

        if filters:
            try:
                qdrant_filter = self._build_qdrant_filter(filters)
            except Exception as exc:
                logger.warning(
                    "Failed to build Qdrant filter, searching without filter: %s",
                    exc,
                )
                qdrant_filter = None

        hits = None

        try:
            # Preferred for older qdrant-client versions.
            if hasattr(self.client, "search"):
                try:
                    hits = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=top_k,
                        query_filter=qdrant_filter,
                        with_payload=True,
                    )
                except Exception as search_exc:
                    # If .search exists but fails for API/version reasons,
                    # try the newer query_points API if available.
                    if hasattr(self.client, "query_points"):
                        logger.debug(
                            "Qdrant client.search failed, falling back to query_points: %s",
                            search_exc,
                        )
                        response = self.client.query_points(
                            collection_name=self.collection_name,
                            query=query_embedding,
                            limit=top_k,
                            query_filter=qdrant_filter,
                            with_payload=True,
                        )
                        hits = getattr(response, "points", response)
                    else:
                        raise
            elif hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                hits = getattr(response, "points", response)
            else:
                raise VectorDatabaseError(
                    "Installed qdrant-client does not expose "
                    "search() or query_points()."
                )

        except VectorDatabaseError:
            raise
        except Exception as exc:
            raise VectorDatabaseError(f"Search failed: {exc}") from exc

        results: List[RetrievedChunk] = []

        for hit in hits or []:
            payload = getattr(hit, "payload", None) or {}

            chunk = self._payload_to_chunk(payload)

            if chunk is None:
                continue

            score = getattr(hit, "score", 0.0)

            try:
                score = float(score)
            except Exception:
                score = 0.0

            score = max(0.0, score)

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    retriever_type="dense",
                )
            )

        return results

    def _build_qdrant_filter(self, metadata_filter: MetadataFilter) -> Filter:
        """Translates abstract MetadataFilter into Qdrant Filter."""
        conditions = []
        for cond in metadata_filter.conditions:
            if cond.operator == "eq":
                conditions.append(FieldCondition(key=cond.field, match=MatchValue(value=cond.value)))
            # Additional operators (gt, lt, etc.) can be added here as needed
        
        if not conditions:
            return None
            
        # 'must' acts as AND, 'should' acts as OR in Qdrant
        if metadata_filter.operator == "and":
            return Filter(must=conditions)
        else:
            return Filter(should=conditions)

    def delete(self, chunk_ids: List[str]) -> None:
        try:
            self.client.delete(collection_name=self.collection_name, points_selector=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} points from '{self.collection_name}'")
        except Exception as e:
            raise VectorDatabaseError(f"Delete failed: {e}")

    def update(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        # Qdrant's upsert handles updates natively
        self.insert(chunks, embeddings)

    def _payload_to_chunk(self, payload: Optional[Dict[str, Any]]) -> Optional[Chunk]:
        """
        Convert Qdrant payload dictionary into a domain Chunk.

        This is intentionally defensive because Qdrant payloads may come
        from different ingestion versions and may contain missing fields.
        """
        if not payload:
            return None

        try:
            document_id_raw = payload.get("document_id")
            chunk_index_raw = payload.get("chunk_index")
            content = payload.get("content", "")

            if document_id_raw is None or chunk_index_raw is None:
                logger.warning(
                    "Skipping Qdrant payload missing document_id or chunk_index."
                )
                return None

            chunk_type_raw = str(payload.get("chunk_type") or "other").strip()

            try:
                chunk_type = ChunkType(chunk_type_raw)
            except ValueError:
                # Fallback for unknown chunk types from older/newer ingestion.
                chunk_type = getattr(ChunkType, "OTHER", list(ChunkType)[0])

            reserved_keys = {
                "document_id",
                "chunk_index",
                "content",
                "chunk_type",
                "article_number",
                "section_title",
            }

            metadata = {
                key: value
                for key, value in payload.items()
                if key not in reserved_keys
            }

            return Chunk(
                document_id=uuid.UUID(str(document_id_raw)),
                chunk_index=int(chunk_index_raw),
                content=str(content),
                chunk_type=chunk_type,
                article_number=payload.get("article_number"),
                section_title=payload.get("section_title"),
                metadata=metadata,
            )

        except Exception as exc:
            logger.warning("Failed to convert Qdrant payload to Chunk: %s", exc)
            return None


    def scroll_all(
        self,
        page_size: int = 256,
        max_points: int = 100000,
    ) -> List[Chunk]:
        """
        Read existing chunks from Qdrant for in-memory BM25 indexing.

        This is used at startup to rebuild the sparse retrieval index.
        BM25 is in-memory, so it must be rebuilt after backend restart.
        """
        chunks: List[Chunk] = []
        offset = None

        try:
            while len(chunks) < max_points:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                # Older qdrant-client returns tuple: (points, next_offset)
                # Newer versions may return an object.
                if isinstance(response, tuple):
                    points, next_offset = response
                else:
                    points = getattr(response, "points", None)
                    if points is None:
                        points = getattr(response, "records", [])
                    next_offset = getattr(response, "next_offset", None)

                for point in points or []:
                    payload = getattr(point, "payload", None) or {}

                    chunk = self._payload_to_chunk(payload)

                    if chunk is not None:
                        chunks.append(chunk)

                if next_offset is None:
                    break

                offset = next_offset

            return chunks

        except Exception as exc:
            raise VectorDatabaseError(
                f"Failed to scroll Qdrant collection: {exc}"
            ) from exc


    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        return self.insert(chunks, embeddings)

    def _create_client(self) -> QdrantClient:
        try:
            if settings.QDRANT_URL.startswith("http"):
                kwargs = {"url": settings.QDRANT_URL}
                
                # Always pass the API key if configured, even for local HTTP.
                # Your local Qdrant server requires it for authentication.
                if settings.QDRANT_API_KEY:
                    kwargs["api_key"] = settings.QDRANT_API_KEY
                
                # Suppress the Qdrant client's "insecure connection" warning
                # since we are intentionally using HTTP for local development.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", 
                        message="Api key is used with an insecure connection"
                    )
                    return QdrantClient(**kwargs)
            else:
                # Local persistent storage (file-based)
                return QdrantClient(path=settings.QDRANT_URL)
        except Exception as e:
            raise VectorDatabaseError(f"Failed to connect to Qdrant: {e}")