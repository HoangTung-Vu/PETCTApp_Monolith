from sqlalchemy import create_engine, inspect, text
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
    _add_missing_columns()


def _add_missing_columns():
    """create_all() never alters an existing table — add new nullable columns."""
    existing = {c["name"] for c in inspect(engine).get_columns("sessions")}
    table = Base.metadata.tables["sessions"]
    with engine.begin() as conn:
        for column in table.columns:
            if column.name not in existing:
                col_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {column.name} {col_type}"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
