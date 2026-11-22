# src/vector_store/collection_manager.py
"""
Manages Qdrant collection lifecycle, creation, and validation.
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.domain.exceptions import VectorDatabaseError

logger = logging.getLogger(__name__)

class CollectionManager:
    """
    Handles safe creation and validation of Qdrant collections.
    Prevents accidental deletion or overwriting of existing collections.
    """
    def __init__(self, client: QdrantClient):
        self.client = client

    def ensure_collection(self, collection_name: str, vector_size: int, distance: Distance):
        """
        Creates the collection if it doesn't exist, or validates its configuration.
        Raises VectorDatabaseError if an existing collection has incompatible settings.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                self._validate_collection(collection_name, vector_size, distance)
            else:
                self._create_collection(collection_name, vector_size, distance)
        except VectorDatabaseError:
            raise
        except Exception as e:
            raise VectorDatabaseError(f"Collection management failed: {e}")

    def _create_collection(self, collection_name: str, vector_size: int, distance: Distance):
        logger.info(f"Creating new Qdrant collection: {collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )

    def _validate_collection(self, collection_name: str, vector_size: int, distance: Distance):
        info = self.client.get_collection(collection_name)
        actual_size = info.config.params.vectors.size
        actual_distance = info.config.params.vectors.distance
        
        if actual_size != vector_size or actual_distance != distance:
            raise VectorDatabaseError(
                f"Collection '{collection_name}' exists but has incompatible configuration. "
                f"Expected size={vector_size}, distance={distance}. "
                f"Found size={actual_size}, distance={actual_distance}. "
                "Please delete the collection manually or use a different name."
            )
        logger.info(f"Collection '{collection_name}' validated successfully.")