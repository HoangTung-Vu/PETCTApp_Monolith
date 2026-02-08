"""Session repository for database operations."""
from typing import Optional, List
from sqlalchemy.orm import Session as DBSession

from .models import Session
from .db import SessionLocal


class SessionRepository:
    """Repository for Session CRUD operations."""
    
    def __init__(self, db: Optional[DBSession] = None):
        self._db = db
        self._owns_session = db is None
    
    @property
    def db(self) -> DBSession:
        if self._db is None:
            self._db = SessionLocal()
        return self._db
    
    def close(self):
        if self._owns_session and self._db:
            self._db.close()
            self._db = None
    
    def create(
        self,
        patient_name: Optional[str] = None,
        doctor_name: Optional[str] = None,
        ct_path: Optional[str] = None,
        pet_path: Optional[str] = None
    ) -> Session:
        """Create a new session."""
        session = Session(
            patient_name=patient_name,
            doctor_name=doctor_name,
            ct_path=ct_path,
            pet_path=pet_path,
            status="active"
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def get_by_id(self, session_id: int) -> Optional[Session]:
        """Get session by ID."""
        return self.db.query(Session).filter(Session.id == session_id).first()
    
    def get_all(self, limit: int = 20) -> List[Session]:
        """Get all sessions, most recent first."""
        return (
            self.db.query(Session)
            .order_by(Session.created_at.desc())
            .limit(limit)
            .all()
        )
    
    def get_active(self) -> List[Session]:
        """Get active sessions."""
        return (
            self.db.query(Session)
            .filter(Session.status == "active")
            .order_by(Session.created_at.desc())
            .all()
        )
    
    def update(
        self,
        session_id: int,
        ct_path: Optional[str] = None,
        pet_path: Optional[str] = None,
        tumor_seg_path: Optional[str] = None,
        organ_seg_path: Optional[str] = None,
        status: Optional[str] = None
    ) -> Optional[Session]:
        """Update session fields."""
        session = self.get_by_id(session_id)
        if not session:
            return None
        
        if ct_path is not None:
            session.ct_path = ct_path
        if pet_path is not None:
            session.pet_path = pet_path
        if tumor_seg_path is not None:
            session.tumor_seg_path = tumor_seg_path
        if organ_seg_path is not None:
            session.organ_seg_path = organ_seg_path
        if status is not None:
            session.status = status
        
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def delete(self, session_id: int) -> bool:
        """Delete a session."""
        session = self.get_by_id(session_id)
        if not session:
            return False
        self.db.delete(session)
        self.db.commit()
        return True
