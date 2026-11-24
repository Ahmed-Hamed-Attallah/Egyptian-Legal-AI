# src/vector_store/qdrant_store.py
"""Qdrant implementation of the vector store.

Filtering contract (SECURITY)
-----------------------------
Access-control filters must FAIL CLOSED. A filter that cannot be translated
raises instead of silently searching the whole collection.

Translation is a best-effort *push-down* for performance. Correctness is
guaranteed by a mandatory Python post-filter using the same
`MetadataFilter.matches()` evaluator as sparse retrieval and SQL, so any
operator Qdrant cannot express natively still yields the right result set.

Payload contract
----------------
`scope` ("official" | "private") and `owner_id` are written on every point.
They are the fields the visibility policy evaluates, so they must stay in sync
with the SQL row — see `set_document_payload`.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any, Dict, List, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector, IsEmptyCondition,
    IsNullCondition, MatchAny, MatchValue, PayloadField, PointIdsList,
    PointStruct, Range,
)

from src.config.settings import settings
from src.domain.chunks import Chunk, ChunkType
from src.domain.exceptions import VectorDatabaseError
from src.domain.queries import RetrievedChunk
from src.retrieval.metadata_filters import FilterCondition, MetadataFilter
from src.vector_store.base import BaseVectorStore
from src.vector_store.collection_manager import CollectionManager

logger = logging.getLogger(__name__)

# Sentinel: this condition cannot be expressed in Qdrant. Omitting it widens
# the candidate set, which the mandatory post-filter then narrows correctly.
_UNPUSHABLE = object()

_RESERVED_PAYLOAD_KEYS = frozenset({
    "document_id", "chunk_index", "content", "chunk_type",
    "article_number", "section_title", "tenant_id", "owner_id", "scope",
    "chunk_id", "law_name", "law_number", "law_year", "page_number",
    "source_path", "storage_key", "vector_id",
})

# Over-fetch factor when a filter is active, so post-filtering still fills top_k.
_OVERSAMPLE = 4
_MAX_FETCH = 512


def _scalar(value: Any) -> Any:
    """Qdrant payload scalars: UUID → str, enums → value, bool/int/str kept."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector storage preserving legal metadata and access boundaries."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.client = client or self._create_client()
        self.collection_manager = CollectionManager(self.client)
        self._ensure_collection()

    # ------------------------------------------------------------------ setup
    def _ensure_collection(self) -> None:
        self.collection_manager.ensure_collection(
            collection_name=self.collection_name,
            vector_size=settings.VECTOR_DIMENSION,
            distance=self._get_distance_metric(),
        )

    def _get_distance_metric(self) -> Distance:
        return {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }.get(settings.DISTANCE_METRIC.lower(), Distance.COSINE)

    def _create_client(self) -> QdrantClient:
        try:
            if settings.QDRANT_URL.startswith("http"):
                kwargs: Dict[str, Any] = {"url": settings.QDRANT_URL}
                if settings.QDRANT_API_KEY:
                    kwargs["api_key"] = settings.QDRANT_API_KEY
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Api key is used with an insecure connection"
                    )
                    return QdrantClient(**kwargs)
            return QdrantClient(path=settings.QDRANT_URL)
        except Exception as exc:
            raise VectorDatabaseError(f"Failed to connect to Qdrant: {exc}") from exc

    # ----------------------------------------------------------------- writes
    def _generate_point_id(self, chunk: Chunk) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.document_id}_{chunk.chunk_index}"))

    def _chunk_to_point(self, chunk: Chunk, embedding: List[float]) -> PointStruct:
        metadata = getattr(chunk, "metadata", None) or {}
        point_id = str(
            metadata.get("vector_id") or metadata.get("point_id") or self._generate_point_id(chunk)
        )

        document_id = getattr(chunk, "document_id", None) or metadata.get("document_id")
        chunk_index = getattr(chunk, "chunk_index", None)
        if chunk_index is None:
            chunk_index = metadata.get("chunk_index", 0)

        chunk_type = getattr(chunk, "chunk_type", None)
        chunk_type = str(chunk_type) if chunk_type is not None else metadata.get("chunk_type")

        payload: Dict[str, Any] = {
            "document_id": str(document_id) if document_id is not None else None,
            "chunk_index": int(chunk_index or 0),
            "content": chunk.content,
            "chunk_type": chunk_type,
            "article_number": getattr(chunk, "article_number", None) or metadata.get("article_number"),
            "section_title": getattr(chunk, "section_title", None) or metadata.get("section_title"),
        }

        for key in (
            "tenant_id", "owner_id", "scope", "chunk_id", "law_name", "law_number",
            "law_year", "page_number", "source_path", "storage_key", "vector_id",
        ):
            if key in metadata and key not in payload:
                payload[key] = _scalar(metadata[key])

        for key, value in metadata.items():
            if key not in _RESERVED_PAYLOAD_KEYS and key not in payload:
                payload[key] = value

        payload = {k: v for k, v in payload.items() if v is not None}
        return PointStruct(id=point_id, vector=embedding, payload=payload)

    def insert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        if len(chunks) != len(embeddings):
            raise VectorDatabaseError("Number of chunks and embeddings must match")
        points = [self._chunk_to_point(c, e) for c, e in zip(chunks, embeddings)]
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info("Inserted/Updated %d points into '%s'", len(points), self.collection_name)
            return [str(p.id) for p in points]
        except Exception as exc:
            raise VectorDatabaseError(f"Failed to insert points: {exc}") from exc

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        return self.insert(chunks, embeddings)

    def update(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        self.insert(chunks, embeddings)

    def set_document_payload(
        self, document_id: str, tenant_id: str, payload: Dict[str, Any]
    ) -> int:
        """Update payload fields on every point of a document.

        Required when scope/owner changes: retrieval filters read the payload,
        so a SQL-only change would leave visibility out of sync and the document
        would silently disappear from (or leak into) search results.

        Returns the number of points targeted.
        """
        clean = {k: _scalar(v) for k, v in payload.items() if v is not None}
        if not clean:
            return 0

        selector = FilterSelector(filter=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        ]))
        try:
            affected = self.count_points(
                MetadataFilter.all_of(
                    FilterCondition("document_id", str(document_id), "eq"),
                    FilterCondition("tenant_id", tenant_id, "eq"),
                )
            )
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=clean,
                points=selector,
                wait=True,
            )
            logger.info(
                "Updated payload %s on %d point(s) for document=%s tenant=%s",
                sorted(clean), affected, document_id, tenant_id,
            )
            return affected
        except Exception as exc:
            raise VectorDatabaseError(f"set_document_payload failed: {exc}") from exc

    # ------------------------------------------------------- filter push-down
    def _translate_condition(self, cond: FilterCondition):
        key, op, value = cond.field, cond.operator, cond.value

        if op == "eq":
            return FieldCondition(key=key, match=MatchValue(value=_scalar(value)))
        if op == "neq":
            return Filter(must_not=[FieldCondition(key=key, match=MatchValue(value=_scalar(value)))])
        if op == "in":
            values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
            return FieldCondition(key=key, match=MatchAny(any=[_scalar(v) for v in values]))
        if op == "nin":
            values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
            return Filter(must_not=[FieldCondition(key=key, match=MatchAny(any=[_scalar(v) for v in values]))])
        if op in {"gt", "gte", "lt", "lte"}:
            try:
                bound = float(value)
            except (TypeError, ValueError):
                return _UNPUSHABLE
            return FieldCondition(key=key, range=Range(**{op: bound}))
        if op == "exists":
            return Filter(must_not=[
                IsEmptyCondition(is_empty=PayloadField(key=key)),
                IsNullCondition(is_null=PayloadField(key=key)),
            ])
        if op == "contains":
            if isinstance(value, (list, tuple, set, frozenset)):
                return FieldCondition(key=key, match=MatchAny(any=[_scalar(v) for v in value]))
            # Substring search needs a full-text index; post-filter handles it.
            return _UNPUSHABLE
        return _UNPUSHABLE

    def _translate(self, metadata_filter: MetadataFilter) -> Optional[Filter]:
        """None means 'no constraint pushed' — post-filter still enforces it."""
        if metadata_filter.is_empty():
            return None

        parts: List[Any] = []
        for cond in metadata_filter.conditions:
            translated = self._translate_condition(cond)
            if translated is _UNPUSHABLE:
                # Dropping a branch of an OR would NARROW results — unsafe.
                if metadata_filter.operator == "or":
                    return None
                continue
            parts.append(translated)

        for group in metadata_filter.groups:
            sub = self._translate(group)
            if sub is None:
                if metadata_filter.operator == "or":
                    return None
                continue
            parts.append(sub)

        if not parts:
            return None
        if metadata_filter.operator == "and":
            return Filter(must=parts)
        return Filter(should=parts)

    def _build_qdrant_filter(self, metadata_filter: MetadataFilter) -> Optional[Filter]:
        """FAIL CLOSED: a translation error must never widen the search."""
        try:
            return self._translate(metadata_filter)
        except Exception as exc:
            logger.error(
                "Filter translation failed for '%s': %s", metadata_filter.describe(), exc
            )
            raise VectorDatabaseError(f"Cannot enforce access filter: {exc}") from exc

    # ----------------------------------------------------------------- search
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[RetrievedChunk]:
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None

        fetch_k = top_k
        if filters is not None and not filters.is_empty():
            fetch_k = min(max(top_k * _OVERSAMPLE, top_k), _MAX_FETCH)

        hits = self._execute_search(query_embedding, fetch_k, qdrant_filter)

        results: List[RetrievedChunk] = []
        rejected = 0
        for hit in hits or []:
            payload = getattr(hit, "payload", None) or {}

            # Mandatory verification: identical evaluator to sparse/SQL.
            if filters is not None and not filters.matches(payload):
                rejected += 1
                continue

            chunk = self._payload_to_chunk(payload)
            if chunk is None:
                continue
            try:
                score = max(0.0, float(getattr(hit, "score", 0.0)))
            except (TypeError, ValueError):
                score = 0.0

            results.append(RetrievedChunk(chunk=chunk, score=score, retriever_type="dense"))
            if len(results) >= top_k:
                break

        if rejected:
            logger.info(
                "DENSE FILTER | dropped=%d | kept=%d | filter=%s",
                rejected, len(results), filters.describe() if filters else "*",
            )
        return results

    def _execute_search(self, query_embedding, limit, qdrant_filter):
        try:
            if hasattr(self.client, "search"):
                try:
                    return self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=limit,
                        query_filter=qdrant_filter,
                        with_payload=True,
                    )
                except Exception as exc:
                    if not hasattr(self.client, "query_points"):
                        raise
                    logger.debug("client.search failed; using query_points: %s", exc)
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                return getattr(response, "points", response)
            raise VectorDatabaseError(
                "Installed qdrant-client exposes neither search() nor query_points()."
            )
        except VectorDatabaseError:
            raise
        except Exception as exc:
            raise VectorDatabaseError(f"Search failed: {exc}") from exc

    # ------------------------------------------------------------------ reads
    def _payload_to_chunk(self, payload: Optional[Dict[str, Any]]) -> Optional[Chunk]:
        if not payload:
            return None
        try:
            document_id_raw = payload.get("document_id")
            chunk_index_raw = payload.get("chunk_index")
            if document_id_raw is None or chunk_index_raw is None:
                logger.warning("Skipping Qdrant payload missing document_id or chunk_index.")
                return None

            try:
                chunk_type = ChunkType(str(payload.get("chunk_type") or "other").strip())
            except ValueError:
                chunk_type = getattr(ChunkType, "OTHER", list(ChunkType)[0])

            metadata = {
                k: v for k, v in payload.items()
                if k not in {"document_id", "chunk_index", "content", "chunk_type",
                             "article_number", "section_title"}
            }
            return Chunk(
                document_id=uuid.UUID(str(document_id_raw)),
                chunk_index=int(chunk_index_raw),
                content=str(payload.get("content", "")),
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
        filters: Optional[MetadataFilter] = None,
    ) -> List[Chunk]:
        """Read chunks for in-memory BM25 indexing."""
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None
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
                    scroll_filter=qdrant_filter,
                )
                if isinstance(response, tuple):
                    points, next_offset = response
                else:
                    points = getattr(response, "points", None) or getattr(response, "records", [])
                    next_offset = getattr(response, "next_offset", None)

                for point in points or []:
                    payload = getattr(point, "payload", None) or {}
                    if filters is not None and not filters.matches(payload):
                        continue
                    chunk = self._payload_to_chunk(payload)
                    if chunk is not None:
                        chunks.append(chunk)

                if next_offset is None:
                    break
                offset = next_offset
            return chunks
        except VectorDatabaseError:
            raise
        except Exception as exc:
            raise VectorDatabaseError(f"Failed to scroll Qdrant collection: {exc}") from exc

    def count_points(self, filters: Optional[MetadataFilter] = None) -> int:
        """Exact point count, optionally scoped. Used for health and reconciliation."""
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=qdrant_filter,
                exact=True,
            )
            return int(getattr(result, "count", 0))
        except Exception as exc:
            raise VectorDatabaseError(f"Count failed: {exc}") from exc

    def list_document_ids(self, max_points: int = 200000) -> Set[str]:
        """Distinct document_ids present in the collection.

        Reconciliation aid: compare against Postgres `documents.id` to find
        orphaned vectors whose SQL row was deleted.
        """
        ids: Set[str] = set()
        offset = None
        scanned = 0
        try:
            while scanned < max_points:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=512,
                    offset=offset,
                    with_payload=["document_id"],
                    with_vectors=False,
                )
                if isinstance(response, tuple):
                    points, next_offset = response
                else:
                    points = getattr(response, "points", None) or getattr(response, "records", [])
                    next_offset = getattr(response, "next_offset", None)

                for point in points or []:
                    payload = getattr(point, "payload", None) or {}
                    doc_id = payload.get("document_id")
                    if doc_id is not None:
                        ids.add(str(doc_id))
                scanned += len(points or [])
                if next_offset is None:
                    break
                offset = next_offset
            return ids
        except Exception as exc:
            raise VectorDatabaseError(f"list_document_ids failed: {exc}") from exc

    # ---------------------------------------------------------------- deletes
    def delete(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=list(chunk_ids)),
            )
            logger.info("Deleted %d points from '%s'", len(chunk_ids), self.collection_name)
        except Exception as exc:
            raise VectorDatabaseError(f"Delete failed: {exc}") from exc

    def delete_by_document(self, document_id: str, tenant_id: str) -> None:
        """Purge every vector of a document. Required when a document is deleted."""
        selector = FilterSelector(filter=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        ]))
        try:
            self.client.delete(
                collection_name=self.collection_name, points_selector=selector, wait=True
            )
            logger.info("Purged vectors for document=%s tenant=%s", document_id, tenant_id)
        except Exception as exc:
            raise VectorDatabaseError(f"Delete by document failed: {exc}") from exc

    def delete_missing_scope(self, tenant_id: Optional[str] = None) -> int:
        """Delete points that carry no `scope` payload.

        Legacy bulk-ingested points predate the ownership model and can never
        satisfy the visibility policy, so they are unreachable dead weight that
        still costs BM25 rebuild time on every startup.
        """
        conditions = [FilterCondition("scope", None, "exists")]
        if tenant_id:
            conditions.append(FilterCondition("tenant_id", tenant_id, "eq"))
        # `exists` is negated here: we want points where scope is absent.
        absent = Filter(must_not=[IsEmptyCondition(is_empty=PayloadField(key="scope"))])
        selector_filter = Filter(must=[
            Filter(must_not=[absent]),
            *([FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))] if tenant_id else []),
        ])
        try:
            before = self.count_points()
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=selector_filter),
                wait=True,
            )
            removed = before - self.count_points()
            logger.warning("Deleted %d point(s) with no scope payload.", removed)
            return removed
        except Exception as exc:
            raise VectorDatabaseError(f"delete_missing_scope failed: {exc}") from exc