from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.storage.database.models.user import UserRecord

class UserRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_email(self, email: str) -> Optional[UserRecord]:
        stmt = select(UserRecord).where(UserRecord.email == email)
        return self._session.execute(stmt).scalar_one_or_none()

    def create(self, user: UserRecord) -> UserRecord:
        self._session.add(user)
        self._session.flush()
        return user