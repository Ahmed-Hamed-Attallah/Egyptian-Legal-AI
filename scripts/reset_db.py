#!/usr/bin/env python3
"""
Reset PostgreSQL database before re-ingestion.

Safely deletes all records from documents, chunks, conversations, and messages
using SQLAlchemy ORM — respects foreign key relationships automatically.

Usage:
    python -m scripts.reset_database
    python -m scripts.reset_database --only-documents
"""
from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy.orm import Session

from src.config.logging_config import setup_logging
from src.storage.database.database import get_database
from src.storage.database.models.chunk import DocumentChunkRecord
from src.storage.database.models.conversation import ConversationRecord, MessageRecord
from src.storage.database.models.document import DocumentRecord

logger = logging.getLogger(__name__)


def reset_documents(session: Session) -> int:
    """Delete all documents and their related chunks/messages."""
    chunk_count = session.query(DocumentChunkRecord).delete()
    message_count = session.query(MessageRecord).delete()
    conversation_count = session.query(ConversationRecord).delete()
    document_count = session.query(DocumentRecord).delete()
    session.commit()
    logger.info(
        "Deleted: %d chunks, %d messages, %d conversations, %d documents",
        chunk_count, message_count, conversation_count, document_count,
    )
    return document_count


def reset_only_documents(session: Session) -> int:
    """Delete only documents (and their chunks), keep conversations."""
    chunk_count = session.query(DocumentChunkRecord).delete()
    document_count = session.query(DocumentRecord).delete()
    session.commit()
    logger.info("Deleted: %d chunks, %d documents", chunk_count, document_count)
    return document_count


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Reset PostgreSQL database.")
    parser.add_argument(
        "--only-documents",
        action="store_true",
        help="Keep conversations and messages; delete only documents and chunks.",
    )
    args = parser.parse_args()

    db = get_database()
    db.create_all()

    with db.session() as session:
        if args.only_documents:
            count = reset_only_documents(session)
        else:
            count = reset_documents(session)

    logger.info("✅ Reset complete. %d document(s) removed.", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())