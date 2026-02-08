from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from .models import Base
import os

# Ensure datadir exists
DB_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'storage')
os.makedirs(DB_FOLDER, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DB_FOLDER, 'petct.db')}"

engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
