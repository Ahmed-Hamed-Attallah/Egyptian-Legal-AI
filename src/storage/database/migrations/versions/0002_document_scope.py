"""Document visibility scope + owner backfill.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-24

Adds `documents.scope` (OFFICIAL | PRIVATE) so the curated corpus is shared
while user uploads stay owner-only (ARCHITECTURE §7). Also reconciles the
`indexed` column, which the ORM declares but 0001 never created.

Idempotent: every step inspects the live schema first, because some
deployments created tables via Base.metadata.create_all().
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("documents"):
        return

    columns = {c["name"] for c in inspector.get_columns("documents")}
    indexes = {i["name"] for i in inspector.get_indexes("documents")}

    if "indexed" not in columns:
        op.add_column(
            "documents",
            sa.Column("indexed", sa.Boolean(), nullable=False, server_default=sa.false()),
        )

    if "scope" not in columns:
        op.add_column(
            "documents",
            sa.Column("scope", sa.String(16), nullable=False, server_default="PRIVATE"),
        )

    if "ix_documents_scope" not in indexes:
        op.create_index("ix_documents_scope", "documents", ["scope"])
    if "ix_documents_tenant_scope_owner" not in indexes:
        op.create_index(
            "ix_documents_tenant_scope_owner", "documents", ["tenant_id", "scope", "owner_id"]
        )

    # Backfill ownership from the job that created each document. Without this,
    # every job-mode upload has owner_id NULL and becomes invisible to its owner.
    if bind.dialect.name == "postgresql" and inspector.has_table("processing_jobs"):
        op.execute(
            """
            UPDATE documents d
            SET owner_id = j.requested_by
            FROM processing_jobs j
            WHERE j.document_id = d.id
              AND d.owner_id IS NULL
              AND j.requested_by IS NOT NULL
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("documents"):
        return

    indexes = {i["name"] for i in inspector.get_indexes("documents")}
    columns = {c["name"] for c in inspector.get_columns("documents")}

    if "ix_documents_tenant_scope_owner" in indexes:
        op.drop_index("ix_documents_tenant_scope_owner", table_name="documents")
    if "ix_documents_scope" in indexes:
        op.drop_index("ix_documents_scope", table_name="documents")
    if "scope" in columns:
        op.drop_column("documents", "scope")
    # `indexed` is intentionally kept: the ORM depends on it.