from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()

class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"

class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_name = Column(String, nullable=True)
    doctor_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    pet_path = Column(String, nullable=True)
    ct_path = Column(String, nullable=True)
    tumor_seg_path = Column(String, nullable=True)
    organ_seg_path = Column(String, nullable=True)
    # DICOM sessions: ct_path/pet_path hold the series folders, and the series
    # are re-located by UID on reload. NULL for NIfTI sessions.
    ct_series_uid = Column(String, nullable=True)
    pet_series_uid = Column(String, nullable=True)
    resample_mode = Column(String, nullable=True)   # "ct" | "pet" grid
    status = Column(String, default="active") # Simple string for now to avoid enum complexity with sqlite if not handled carefully, or use Enum type.

    def __repr__(self):
        return f"<Session(id={self.id}, patient={self.patient_name})>"
