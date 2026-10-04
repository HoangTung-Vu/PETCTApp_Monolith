from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, scoped_session
from .models import Base
from ..core.config import settings

# The database location is chosen by the user at startup, so the engine is
# created by init_db() rather than at import. SessionLocal is bound there too;
# modules that imported it earlier hold the same object and see the binding.
engine = None

SessionLocal = sessionmaker(autocommit=False, autoflush=False)

def init_db(db_path=None):
    """Open (creating if needed) the database at ``db_path`` or ``settings.DB_PATH``."""
    global engine
    path = Path(db_path or settings.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if engine is not None:
        engine.dispose()
    engine = create_engine(f"sqlite:///{path}", echo=False)
    SessionLocal.configure(bind=engine)

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
