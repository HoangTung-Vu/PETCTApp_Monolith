# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Full launch (builds Docker AI engines, starts containers, launches GUI):
./start.sh          # Linux/WSL
.\start.bat         # Windows

# GUI only (requires Docker engines already running separately):
uv run python -m src.main
```

## Running Tests

```bash
# All tests:
uv run pytest tests/

# Single test file:
uv run pytest tests/test_prob_system.py

# Quick import verification (no Docker needed):
uv run python tests/verify_gui_imports.py
uv run python tests/verify_refinement_logic.py
```

## Architecture Overview

**Monolith GUI + 3 dockerized AI backends communicating via HTTP.**

- GUI host: PyQt6 + embedded Napari viewers
- AI engines (each a FastAPI Docker container, ports in `.env`):
  - nnUNet → port 8101 — automated segmentation
  - AutoPET → port 8102 — interactive probability-based segmentation
  - TotalSegmentator → port 8103 — anatomy segmentation

### GUI Layer (`src/gui/`)

`MainWindow` is a **slim coordinator** (~150 lines) that inherits from 5 handler mixins:

```
MainWindow
├── SegmentationHandlerMixin  → load → run engine → save workflow
├── RefinementHandlerMixin    → SUV / iterative thresholding
├── AutoPETHandlerMixin       → interactive click-based refinement
├── EraserHandlerMixin        → paint/erase + undo stack
└── ReportHandlerMixin        → clinical metric report generation
```

**ControlPanel** (`components/control_panel.py`) is the signal hub — it owns the left sidebar tabs and emits Qt signals that `MainWindow` connects to handler methods.

**LayoutManager** (`components/layout/layout_manager.py`) manages three Napari viewer layouts: Orthogonal (CT/PET/mask grid), Fusion (overlaid), and 3D.

All heavy operations run in **QThread workers** (`workers/`). A global `_inference_lock` in each worker prevents concurrent HTTP inference requests.

### Core Layer (`src/core/`)

- `session_manager.py` — single source of truth for in-memory session state (CT/PET arrays, mask, probability map, metadata)
- `file_manager.py` — NIfTI/NPZ I/O to `storage/data/<session_id>/`
- `engine/` — HTTP client wrappers for each Docker backend; `refinement_engine.py` and `iterative_thresholding_refinement.py` run locally (no Docker)

### Database (`src/database/`)

SQLAlchemy + SQLite at `storage/petct.db`. Stores session metadata (file paths, timestamps). CRUD via `session_repository.py`.

## Coordinate Spaces

Three different coordinate conventions are used — always convert explicitly:

| Space | Axis order | Used by |
|-------|-----------|---------|
| NIfTI / session_manager | X, Y, Z | nibabel, storage |
| Napari / display | Z, Y, X | viewers, painting |
| SimpleITK | Z, Y, X | AutoPET backend |

Converters in `src/utils/dimension_utils.py`: `to_napari()`, `from_napari()`, `point_from_napari()`.

## Known Performance Issues

1. **Eraser undo stack** (`handlers/eraser_handler.py`): stores full mask+prob copies (~320 MB/entry × 5 entries = ~1.6 GB). Fix: store diffs (indices + values).
2. **Snapshot lag 2–3s** (`workers/snapshot_worker.py`, `handlers/refinement_handler.py`): copies full float32 array on every Refine/AutoPET tab switch. Fix: dirty flag + uint8 storage.
3. **View switch lag** (`components/layout/layout_manager.py`): creates viewers lazily on first switch. Fix: pre-create all viewers at startup.
4. **Load session lag**: Napari `add_image()` runs on the main thread.

## Key Dependencies

- Python 3.13 (managed via pyenv, `.python-version`)
- Package manager: `uv` (see `pyproject.toml` / `uv.lock`)
- `napari[all,optional,pyqt6]` ≥ 0.6.6 — embedded medical image viewer
- `nibabel` — NIfTI file I/O
- `httpx` — async HTTP calls to Docker engines
- `sqlalchemy` — SQLite session persistence
