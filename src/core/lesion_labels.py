"""Lesion ids in the tumor mask.

A tumor mask is either

* **binary** — every non-zero voxel is tumour, and a lesion is a 26-connected
  component. Lesion ids are not stored; they are numbered head → feet
  (``binary_components``), exactly as ``give_lesion_ids`` would write them.
* **labelled** — every voxel holds its lesion id (1..32767, int16 on disk).
  A lesion is all voxels with one id, even when they form several components
  (the user merged them on purpose).

Everything here is plain numpy and keeps the app's memory-layout rule: XYZ
volumes are Fortran-ordered, and scipy / boolean indexing go through the
C-contiguous transpose (30–100× faster than walking an F array in C order).
"""

import numpy as np
from scipy.ndimage import find_objects, label

from ..utils.nifti_utils import MASK_DTYPE, mask_bbox

MAX_LABEL = int(np.iinfo(MASK_DTYPE).max)

# A lesion is a 26-connected component — the same unit a double-click island erase removes
# (skimage.flood's default full connectivity).
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)

# NIfTI intent written into labelled masks, so a mask whose lesions were all
# merged into id 1 is still read back as labelled (values alone say binary).
_INTENT_NAME = "lesion_ids"


def _is_f_like(a: np.ndarray) -> bool:
    """True when the first axis varies fastest in memory (F-ordered volume or crop)."""
    return a.ndim > 1 and a.strides[0] < a.strides[-1]


def _label_components(mask: np.ndarray, structure=None):
    """``scipy.ndimage.label`` + ``find_objects`` in the fast memory order.

    Both are 50–100× slower on Fortran-ordered input (NIfTI's layout), so an
    F-ordered mask is processed through its C-contiguous transpose; labels come
    back in ``mask``'s axis order. Returns ``(labels, n, slices)``.
    """
    if mask.flags.f_contiguous and not mask.flags.c_contiguous:
        st = None if structure is None else structure.T
        labels_t, n = label(mask.T, structure=st)
        slices = [None if s is None else s[::-1] for s in find_objects(labels_t)]
        return labels_t.T, n, slices
    labels, n = label(mask, structure=structure)
    return labels, n, find_objects(labels)


def _find_objects(crop: np.ndarray) -> list:
    """``find_objects`` of an integer crop, through its transpose when F-like."""
    if _is_f_like(crop):
        return [None if s is None else s[::-1] for s in find_objects(crop.T)]
    return find_objects(crop)


def _offset(bb, sl) -> tuple:
    return tuple(slice(o.start + s.start, o.start + s.stop) for o, s in zip(bb, sl))


def _head_to_feet(items: list) -> list:
    """Sort ``(…, abs_slices, …)`` items by bbox z-centre, head first.

    In LAS a larger z is superior. The sort is stable, so ties keep scipy's
    labelling order.
    """
    return sorted(items, key=lambda it: (it[1][2].start + it[1][2].stop - 1) / 2.0, reverse=True)


def _binary_components(mask: np.ndarray):
    """``(bb, labels, comps)``: the non-zero voxels of ``mask`` split into
    26-connected components inside its bounding box ``bb``.

    ``comps`` is ``[(k, abs_slices, bool_crop)]`` ordered head → feet, where
    ``k`` is the component's value in ``labels`` (scipy's numbering).
    """
    bb = mask_bbox(mask)
    if bb is None:
        return None, None, []
    # `>` keeps the crop's F layout, so the transpose fast path applies.
    labels, _, comp_slices = _label_components(mask[bb] > 0, _CONNECTIVITY_26)
    comps = [(k, _offset(bb, sl), labels[sl] == k)
             for k, sl in enumerate(comp_slices, start=1) if sl is not None]
    return bb, labels, _head_to_feet(comps)


def binary_components(mask: np.ndarray) -> list:
    """Lesions of a binary reading of ``mask``: ``[(id, abs_slices, bool_crop)]``.

    Every non-zero voxel counts; 26-connected components get ids 1..N from
    head to feet. ``give_lesion_ids`` writes exactly these ids.
    """
    _, _, comps = _binary_components(mask)
    return [(i, sl, comp) for i, (_, sl, comp) in enumerate(comps, start=1)]


def label_components(mask: np.ndarray) -> list:
    """Lesions of a labelled ``mask``: ``[(id, abs_slices, bool_crop)]`` by id.

    The id is the voxel value. Ids may have gaps, and one id may cover several
    components — it is still one lesion.
    """
    bb = mask_bbox(mask)
    if bb is None:
        return []
    crop = mask[bb]
    if crop.min() < 0:
        raise ValueError("The tumor mask has negative values.")
    out = []
    for lesion_id, sl in enumerate(_find_objects(crop), start=1):
        if sl is not None:
            out.append((lesion_id, _offset(bb, sl), crop[sl] == lesion_id))
    return out


def give_lesion_ids(mask: np.ndarray) -> tuple:
    """Relabel ``mask`` from scratch: binarise, 26-connected components, ids
    1..N head → feet. Returns ``(labelled int16 F-ordered volume, N)``.
    """
    out = np.zeros(mask.shape, dtype=MASK_DTYPE, order="F")
    bb, labels, comps = _binary_components(mask)
    if bb is None:
        return out, 0
    if len(comps) > MAX_LABEL:
        raise ValueError(f"{len(comps)} lesions — more than the {MAX_LABEL} ids a mask can hold.")
    lut = np.zeros(int(labels.max()) + 1, dtype=MASK_DTYPE)
    for new_id, (k, _, _) in enumerate(comps, start=1):
        lut[k] = new_id
    # Index through the C-contiguous transpose of an F-ordered label crop.
    out[bb] = lut[labels.T].T if _is_f_like(labels) else lut[labels]
    return out, len(comps)


def present_labels(mask: np.ndarray) -> list:
    """Sorted non-zero ids present in ``mask``."""
    bb = mask_bbox(mask)
    if bb is None:
        return []
    crop = mask[bb]
    flat = crop.T.ravel() if _is_f_like(crop) else crop.ravel()
    counts = np.bincount(flat.astype(np.intp, copy=False))
    return [int(i) for i in np.flatnonzero(counts) if i > 0]


def overwritten_labels(atoms, new_label: int) -> list:
    """Ids other than background and ``new_label`` that a napari paint
    history item ``[(indices, old_values, new_value), …]`` painted over."""
    found = set()
    for _, old, _ in atoms:
        old = np.asarray(old)
        if old.size:
            vals = np.unique(old)
            found.update(int(v) for v in vals if v != 0 and v != new_label)
    return sorted(found)


def resolve_overwrite(data: np.ndarray, atoms, new_label: int, merge: bool) -> list:
    """Settle a paint stroke of ``new_label`` that covered other lesions.

    ``merge`` — the covered lesions are the same lesion: every voxel of each
    covered id becomes ``new_label``. Otherwise the covered voxels get their
    old id back, so the stroke only painted background. ``data`` is the
    edited array (any layout) and is changed in place. Returns the covered ids.
    """
    covered = overwritten_labels(atoms, new_label)
    if not covered:
        return covered
    if merge:
        work = data.T if _is_f_like(data) else data
        for lesion_id in covered:
            work[work == lesion_id] = new_label
        return covered
    for indices, old, _ in atoms:
        old = np.asarray(old)
        if not old.size:
            continue
        keep = (old != 0) & (old != new_label)
        if keep.any():
            data[tuple(np.asarray(i)[keep] for i in indices)] = old[keep]
    return covered


def roi_conflicts(tumor: np.ndarray, roi: np.ndarray, label_id: int) -> list:
    """Ids other than ``label_id`` lying under the ROI's non-zero voxels."""
    bb = mask_bbox(roi)
    if bb is None:
        return []
    t, r = tumor[bb], roi[bb]
    if _is_f_like(t):
        t, r = t.T, r.T
    vals = np.unique(t[r > 0])
    return [int(v) for v in vals if v != 0 and v != label_id]


def merge_roi(tumor: np.ndarray, roi: np.ndarray, label_id: int, merge_ids=()) -> np.ndarray:
    """Merge a confirmed ROI into ``tumor`` as lesion ``label_id`` (in place).

    ``merge_ids`` lesions are relabelled ``label_id`` first (the user said they
    are the same lesion); then the ROI fills only background voxels, so other
    lesions under it keep their ids. On a binary mask with ``label_id`` 1 this
    is the old ``np.maximum(tumor, roi)``.
    """
    if merge_ids:
        bb = mask_bbox(tumor)
        if bb is not None:
            t = tumor[bb]
            t = t.T if _is_f_like(t) else t
            for lesion_id in merge_ids:
                t[t == lesion_id] = label_id
    bb = mask_bbox(roi)
    if bb is None:
        return tumor
    t, r = tumor[bb], roi[bb]
    if _is_f_like(t):
        t, r = t.T, r.T
    t[(r > 0) & (t == 0)] = label_id
    return tumor


def mark_lesion_ids(header) -> None:
    """Tag a NIfTI header as a labelled lesion mask."""
    header.set_intent("label", (), name=_INTENT_NAME)


def has_lesion_ids_marker(header) -> bool:
    try:
        code, _, name = header.get_intent()
    except Exception:
        return False
    return code == "label" and name == _INTENT_NAME
