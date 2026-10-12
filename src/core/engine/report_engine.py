"""Engine for computing PET/CT clinical report metrics (per-lesion)."""
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

from ...utils.dimension_utils import get_spacing_from_affine, get_voxel_volume_from_affine
from ...utils.nifti_utils import crop_to_napari, mask_bbox


# A lesion is a 26-connected component — the same unit the Eraser removes
# (skimage.flood's default full connectivity).
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)

# SUVpeak sphere: 1 cm³ = 1000 mm³ → r = (3·1000 / 4π)^(1/3) ≈ 6.20 mm.
SUVPEAK_RADIUS_MM = (3.0 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)


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


def _sphere_kernel(spacing_xyz, radius_mm: float):
    """Boolean kernel of the voxel offsets whose centres lie within ``radius_mm``.

    Returns ``(kernel, half)`` where ``half[d]`` is the kernel half-width (voxels)
    along axis ``d``; the kernel is centred at index ``half``.
    """
    half = [int(np.floor(radius_mm / s)) if s > 0 else 0 for s in spacing_xyz]
    ox, oy, oz = np.ogrid[-half[0]:half[0] + 1, -half[1]:half[1] + 1, -half[2]:half[2] + 1]
    sx, sy, sz = spacing_xyz
    kernel = (ox * sx) ** 2 + (oy * sy) ** 2 + (oz * sz) ** 2 <= radius_mm ** 2
    return kernel, half


def _sphere_mean(pet: np.ndarray, center, kernel: np.ndarray, half) -> float:
    """Mean of ``pet`` under ``kernel`` centred on voxel ``center``.

    The sphere is clipped at the volume edges; voxels inside it count whether
    or not they belong to the lesion.
    """
    win, ker = [], []
    for c, h, n in zip(center, half, pet.shape):
        lo, hi = max(0, c - h), min(n, c + h + 1)
        win.append(slice(lo, hi))
        ker.append(slice(lo - (c - h), hi - (c - h)))
    vals = pet[tuple(win)][kernel[tuple(ker)]]
    return float(np.mean(vals, dtype=np.float64))


class ReportEngine:
    """Computes clinical PET metrics on a per-lesion basis using connected
    component analysis.

    Per-lesion metrics:
        MTV     Metabolic Tumour Volume (mL) of the lesion.
        SUVmax  Maximum PET intensity within the lesion.
        SUVmean Mean PET intensity within the lesion.
        SUVpeak Mean PET intensity in a 1 cm³ sphere centred on the SUVmax voxel
                (the sphere is not restricted to the lesion).
        TLG     Total Lesion Glycolysis = SUVmean × MTV.

    Global metrics:
        tTLG    Total TLG = Σ TLG_i.
    """

    @staticmethod
    def compute_lesion_stats(pet: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> dict:
        """Per-lesion quantification of a tumour mask.

        Args:
            pet:    PET volume in SUV (XYZ, LAS), any dtype / memory order.
            mask:   Tumour mask (XYZ, same grid); every non-zero voxel counts.
            affine: 4x4 voxel → mm affine of the grid.

        Returns:
            dict with ``tTLG`` (float) and ``lesions``, a list sorted head → feet
            with 1-based ``id``. Each lesion holds ``MTV`` (mL), ``SUVmax``,
            ``SUVmean``, ``SUVpeak``, ``TLG``, ``n_voxels``, ``bbox``
            (inclusive ``(x0, y0, z0, x1, y1, z1)``), ``slices`` (the bbox as XYZ
            slices), ``mask`` (bool crop of the lesion inside ``slices``),
            ``suvmax_voxel``, ``center_voxel`` (bbox centre, XYZ) and
            ``center_physical`` (mm). Values are unrounded.
        """
        if pet.shape != mask.shape:
            raise ValueError(f"Shape mismatch: PET {pet.shape} vs Mask {mask.shape}")

        bb = mask_bbox(mask)
        if bb is None:
            return {"tTLG": 0.0, "lesions": []}

        voxel_ml = get_voxel_volume_from_affine(affine) / 1000.0
        spacing = [abs(float(s)) for s in get_spacing_from_affine(affine)]
        kernel, half = _sphere_kernel(spacing, SUVPEAK_RADIUS_MM)

        # Label only the mask's bounding box. `>` keeps the crop's F layout, so
        # the transpose fast path of _label_components applies.
        labels, _, comp_slices = _label_components(mask[bb] > 0, _CONNECTIVITY_26)
        offset = [s.start for s in bb]

        lesions = []
        for k, sl in enumerate(comp_slices, start=1):
            if sl is None:
                continue
            comp = labels[sl] == k
            abs_sl = tuple(slice(o + s.start, o + s.stop) for o, s in zip(offset, sl))

            # Boolean indexing / argmax walk the array in C order; on the
            # F-ordered crops that is ~30× slower than going through the
            # transposes (memory order). Element order doesn't matter here.
            comp_t = comp.T
            pet_t = pet[abs_sl].T
            vals = pet_t[comp_t].astype(np.float64)
            n_voxels = int(vals.size)
            suv_mean = float(vals.mean())
            mtv = n_voxels * voxel_ml

            peak_t = np.unravel_index(int(np.argmax(np.where(comp_t, pet_t, -np.inf))), comp_t.shape)
            suvmax_voxel = tuple(int(s.start + i) for s, i in zip(abs_sl, peak_t[::-1]))
            suv_max = float(pet[suvmax_voxel])

            x0, y0, z0 = (s.start for s in abs_sl)
            x1, y1, z1 = (s.stop - 1 for s in abs_sl)
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
            phys = affine @ np.array([*center, 1.0])

            lesions.append({
                "MTV": mtv,
                "SUVmax": suv_max,
                "SUVmean": suv_mean,
                "SUVpeak": _sphere_mean(pet, suvmax_voxel, kernel, half),
                "TLG": suv_mean * mtv,
                "n_voxels": n_voxels,
                "bbox": (x0, y0, z0, x1, y1, z1),
                "slices": abs_sl,
                "mask": comp,
                "suvmax_voxel": suvmax_voxel,
                "center_voxel": center,
                "center_physical": tuple(float(v) for v in phys[:3]),
            })

        # IDs go from head to feet: in LAS, larger z is superior.
        lesions.sort(key=lambda l: (l["bbox"][2] + l["bbox"][5]) / 2.0, reverse=True)
        for idx, lesion in enumerate(lesions, start=1):
            lesion["id"] = idx

        return {
            "tTLG": float(sum(l["TLG"] for l in lesions)),
            "lesions": lesions,
        }

    # ── Report Export ────────────────────────────────────────────────────

    _cmap_lut_cache: dict = {}

    @staticmethod
    def _get_colormap_lut(name: str) -> np.ndarray:
        """Build a 256x3 float32 LUT for a napari colormap name. Cached."""
        if name in ReportEngine._cmap_lut_cache:
            return ReportEngine._cmap_lut_cache[name]

        from napari.utils.colormaps.colormap_utils import ensure_colormap

        cm = ensure_colormap(name)
        t = np.linspace(0.0, 1.0, 256).astype(np.float32)
        rgba = cm.map(t)  # (256, 4)
        lut = rgba[:, :3].astype(np.float32)  # drop alpha
        ReportEngine._cmap_lut_cache[name] = lut
        return lut

    @staticmethod
    def _apply_colormap(data_2d: np.ndarray, window: float, level: float,
                        colormap: str) -> np.ndarray:
        """Window/level + colormap -> (H, W, 3) float32 in [0, 1]."""
        vmin = level - window / 2.0
        vmax = level + window / 2.0
        normed = np.clip((data_2d - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
        idx = (normed * 255).astype(np.intp)
        lut = ReportEngine._get_colormap_lut(colormap)
        return lut[idx]  # (H, W, 3)

    @staticmethod
    def _render_slice(
        vol_data: np.ndarray,
        mask_slice: np.ndarray,
        window: float,
        level: float,
        colormap: str,
        mask_opacity: float,
    ) -> Image.Image:
        """Render a 2D slice with optional tumor mask overlay. Returns PIL Image (RGB)."""
        rgb = ReportEngine._apply_colormap(vol_data, window, level, colormap)

        # Overlay tumor mask in red
        if mask_slice is not None and np.any(mask_slice):
            mask_bool = mask_slice > 0
            overlay_color = np.array([1.0, 0.2, 0.1], dtype=np.float32)
            for ch in range(3):
                rgb[:, :, ch] = np.where(
                    mask_bool,
                    rgb[:, :, ch] * (1 - mask_opacity) + overlay_color[ch] * mask_opacity,
                    rgb[:, :, ch],
                )

        img_arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    _font_cache: dict = {}

    @staticmethod
    def _resolve_font(size: int):
        """Return a PIL font of roughly ``size`` px (cached), with fallbacks."""
        if size in ReportEngine._font_cache:
            return ReportEngine._font_cache[size]
        from PIL import ImageFont

        font = None
        for name in ("DejaVuSans.ttf", "Arial.ttf"):
            try:
                font = ImageFont.truetype(name, size)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size)   # Pillow >= 10.1
            except Exception:
                font = ImageFont.load_default()
        ReportEngine._font_cache[size] = font
        return font

    @staticmethod
    def _draw_ruler_marker(draw, pt, color, r: int):
        """Filled dot + crosshair ticks, mirroring RulerOverlay._draw_marker."""
        x, y = int(round(pt[0])), int(round(pt[1]))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
        inner = int(round(r * 1.4))
        outer = int(round(r * 2.4))
        w = max(1, r // 3)
        draw.line([(x - outer, y), (x - inner, y)], fill=color, width=w)
        draw.line([(x + inner, y), (x + outer, y)], fill=color, width=w)
        draw.line([(x, y - outer), (x, y - inner)], fill=color, width=w)
        draw.line([(x, y + inner), (x, y + outer)], fill=color, width=w)

    @staticmethod
    def export_line_images(
        out_dir,
        ct_napari: np.ndarray,
        pet_napari: np.ndarray,
        mask_napari,
        p1_zyx,
        p2_zyx,
        spacing: tuple,
        ct_wl: tuple,
        pet_wl: tuple,
        ct_colormap: str = "gray",
        pet_colormap: str = "jet",
        mask_opacity: float = 0.5,
        length_mm: float = None
    ):
        """Render 6 images (axial/coronal/sagittal × CT/PET) for the line p1->p2.

        CT uses the midpoint slice, PET uses a full-volume MIP, and the two
        endpoints are joined by a green line with crosshair markers + a yellow
        distance label — matching the live RulerOverlay. ``mask_napari`` may be
        None to skip the overlay. All points are in Napari (Z, Y, X) data space.

        Args:
            out_dir: Output directory (created if missing).
            ct_napari: CT volume in Napari (Z, Y, X) order.
            pet_napari: PET volume in Napari (Z, Y, X) order.
            mask_napari: Mask volume in Napari (Z, Y, X) order, or None to skip.
            p1_zyx: First endpoint [z, y, x] in Napari data space.
            p2_zyx: Second endpoint [z, y, x] in Napari data space.
            spacing: (sx, sy, sz) mm/voxel for aspect-ratio correction.
            ct_wl: (window, level) for CT display.
            pet_wl: (window, level) for PET display.
            ct_colormap: Napari colormap name for CT.
            pet_colormap: Napari colormap name for PET.
            mask_opacity: Mask overlay opacity (0.0-1.0).
            length_mm: If given, draw this length (mm) as a text label at the
                line midpoint of every image. None skips the label.
        """
        from PIL import ImageDraw

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sx, sy, sz = spacing
        ct_w, ct_l = ct_wl
        pet_w, pet_l = pet_wl

        z1, y1, x1 = p1_zyx
        z2, y2, x2 = p2_zyx

        mid_z = max(0, min(int(round((z1 + z2) / 2)), ct_napari.shape[0] - 1))
        mid_y = max(0, min(int(round((y1 + y2) / 2)), ct_napari.shape[1] - 1))
        mid_x = max(0, min(int(round((x1 + x2) / 2)), ct_napari.shape[2] - 1))

        # Use MIP for PET (and mask), midpoint slice for CT.
        pet_axial_mip = np.max(pet_napari, axis=0)
        pet_coronal_mip = np.max(pet_napari, axis=1)
        pet_sagittal_mip = np.max(pet_napari, axis=2)

        if mask_napari is not None:
            mask_axial_mip = np.max(mask_napari, axis=0)
            mask_coronal_mip = np.max(mask_napari, axis=1)
            mask_sagittal_mip = np.max(mask_napari, axis=2)
        else:
            mask_axial_mip = mask_coronal_mip = mask_sagittal_mip = None

        line_slices = {
            "axial": (ct_napari[mid_z, :, :], pet_axial_mip, mask_axial_mip, sx, sy, (x1, y1), (x2, y2)),
            "coronal": (ct_napari[:, mid_y, :], pet_coronal_mip, mask_coronal_mip, sx, sz, (x1, z1), (x2, z2)),
            "sagittal": (ct_napari[:, :, mid_x], pet_sagittal_mip, mask_sagittal_mip, sy, sz, (y1, z1), (y2, z2)),
        }

        for plane, (ct_sl, pet_sl, mask_sl, step_w, step_h, p1, p2) in line_slices.items():
            ct_img = ReportEngine._render_slice(ct_sl, mask_sl, ct_w, ct_l, ct_colormap, mask_opacity)
            pet_img = ReportEngine._render_slice(pet_sl, mask_sl, pet_w, pet_l, pet_colormap, mask_opacity)

            base_w, base_h = ct_img.size
            min_step = min(step_w, step_h)
            new_w, new_h = base_w, base_h
            if min_step > 0:
                new_w = int(round(base_w * (step_w / min_step)))
                new_h = int(round(base_h * (step_h / min_step)))
                if new_w != base_w or new_h != base_h:
                    resample_filter = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                    ct_img = ct_img.resize((new_w, new_h), resample_filter)
                    pet_img = pet_img.resize((new_w, new_h), resample_filter)

            # Scale coordinates to the resized image
            p1_scaled = (p1[0] * (new_w / base_w), p1[1] * (new_h / base_h))
            p2_scaled = (p2[0] * (new_w / base_w), p2[1] * (new_h / base_h))

            # Mirror the live RulerOverlay: green line + endpoint markers +
            # a yellow distance label (with a 1px black shadow for legibility).
            line_color = (0, 255, 128)
            text_color = (255, 255, 0)
            base = min(new_w, new_h)
            line_w = max(2, int(round(base * 0.006)))
            marker_r = max(3, int(round(base * 0.012)))

            label = f"{length_mm:.1f} mm" if length_mm is not None else None
            font = ReportEngine._resolve_font(max(9, int(round(base * 0.025))))
            mx = (p1_scaled[0] + p2_scaled[0]) / 2.0
            my = (p1_scaled[1] + p2_scaled[1]) / 2.0

            for img in (ct_img, pet_img):
                draw = ImageDraw.Draw(img)
                draw.line([p1_scaled, p2_scaled], fill=line_color, width=line_w)
                if label is not None:
                    pad = marker_r + 4
                    try:
                        bbox = draw.textbbox((0, 0), label, font=font)
                        th = bbox[3] - bbox[1]
                    except Exception:
                        th = getattr(font, "size", 12)
                    tx, ty = mx + pad, my - pad - th
                    draw.text((tx + 1, ty + 1), label, font=font, fill=(0, 0, 0))
                    draw.text((tx, ty), label, font=font, fill=text_color)

            ct_img.save(out_dir / f"{plane}_ct.png")
            pet_img.save(out_dir / f"{plane}_pet.png")

    @staticmethod
    def _lesion_plane(lo, crop, shape_zyx, axis: int, index: int) -> np.ndarray:
        """2D bool mask of one lesion on Napari slice ``index`` along ``axis``.

        ``lo`` / ``crop`` are the lesion's ZYX crop (``crop_to_napari``).
        """
        dims = [d for d in range(3) if d != axis]
        out = np.zeros((shape_zyx[dims[0]], shape_zyx[dims[1]]), dtype=bool)
        k = index - lo[axis]
        if 0 <= k < crop.shape[axis]:
            sub = np.take(crop, k, axis=axis)
            a, b = lo[dims[0]], lo[dims[1]]
            out[a:a + sub.shape[0], b:b + sub.shape[1]] = sub
        return out

    @staticmethod
    def export_report(
        report_dir: Path,
        stats: dict,
        ct_zyx: np.ndarray,
        pet_zyx: np.ndarray,
        affine: np.ndarray,
        ct_wl: tuple,
        pet_wl: tuple,
        ct_colormap: str = "gray",
        pet_colormap: str = "jet",
        mask_opacity: float = 0.5,
    ):
        """Export report CSVs and per-tumor slice images from computed lesion stats.

        Wipes and recreates ``report_dir`` on every call. Nothing is recomputed:
        values, IDs and lesion masks all come from ``stats``.

        Args:
            report_dir: Target directory.
            stats: Dict returned by ``compute_lesion_stats``.
            ct_zyx: CT volume in Napari (Z, Y, X) order.
            pet_zyx: PET volume in Napari (Z, Y, X) order (same grid).
            affine: 4x4 NIfTI affine of the grid (XYZ voxel → mm).
            ct_wl: (window, level) for CT display.
            pet_wl: (window, level) for PET display.
            ct_colormap: Napari colormap name for CT.
            pet_colormap: Napari colormap name for PET.
            mask_opacity: Mask overlay opacity (0.0-1.0).
        """
        # Overwrite old report directory
        if report_dir.exists():
            shutil.rmtree(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        sx, sy, sz = (abs(float(s)) for s in get_spacing_from_affine(affine))
        shape_zyx = pet_zyx.shape
        shape_xyz = shape_zyx[::-1]
        shape_x, shape_y, shape_z = shape_xyz

        lesions = stats.get("lesions", [])
        t_tlg = stats.get("tTLG", 0.0)

        def app_coords(voxel_xyz):
            """NIfTI XYZ voxel → Napari (App GUI) (z, y, x)."""
            vx, vy, vz = voxel_xyz
            return shape_z - 1 - vz, shape_y - 1 - vy, vx

        header = [
            "id",
            "voxel_x", "voxel_y", "voxel_z",
            "physical_x_mm", "physical_y_mm", "physical_z_mm",
            "SUVmax", "SUVmean", "SUVpeak", "MTV_mL", "TLG",
        ]

        def lesion_row(les):
            app_z, app_y, app_x = app_coords(les["center_voxel"])
            px, py, pz = les["center_physical"]
            return [
                les["id"],
                int(round(app_x)), int(round(app_y)), int(round(app_z)),
                round(px, 2), round(py, 2), round(pz, 2),
                round(les["SUVmax"], 2), round(les["SUVmean"], 2), round(les["SUVpeak"], 2),
                round(les["MTV"], 2), round(les["TLG"], 2),
            ]

        # ── CSV ──
        csv_path = report_dir / "report_metabolic.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for les in lesions:
                writer.writerow(lesion_row(les))
            # Summary row
            writer.writerow([])
            writer.writerow(["tTLG", round(t_tlg, 2)])

        # ── Additional Metrics (report_radiomics.csv) ──
        d_max = 0.0
        lesion_a, lesion_b = None, None
        if len(lesions) >= 2:
            pts = np.array([les["center_physical"] for les in lesions])
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            idx_a, idx_b = np.unravel_index(np.argmax(dists), dists.shape)
            d_max = round(float(dists[idx_a, idx_b]), 2)
            lesion_a = lesions[idx_a]
            lesion_b = lesions[idx_b]

        report2_path = report_dir / "report_radiomics.csv"
        with open(report2_path, "w", newline="") as f2:
            writer2 = csv.writer(f2)
            writer2.writerow(["Metric", "Value"])
            writer2.writerow(["Dmax_mm", d_max])

            if lesion_a and lesion_b:
                writer2.writerow([])
                writer2.writerow(["Dmax Points Info"])
                writer2.writerow(header)
                for les in (lesion_a, lesion_b):
                    writer2.writerow(lesion_row(les))

        # ── Per-lesion ZYX crops (for isolating each lesion on its slices) ──
        crops = {
            les["id"]: crop_to_napari(les["slices"], les["mask"], shape_xyz)
            for les in lesions
        }

        # ── Per-tumor images ──
        images_dir = report_dir / "images"
        images_dir.mkdir(exist_ok=True)

        ct_w, ct_l = ct_wl
        pet_w, pet_l = pet_wl

        for les in lesions:
            lid = les["id"]
            tumor_dir = images_dir / str(lid)
            tumor_dir.mkdir(exist_ok=True)

            app_z, app_y, app_x = app_coords(les["center_voxel"])
            iz = max(0, min(int(round(app_z)), shape_zyx[0] - 1))
            iy = max(0, min(int(round(app_y)), shape_zyx[1] - 1))
            ix = max(0, min(int(round(app_x)), shape_zyx[2] - 1))

            lo, crop = crops[lid]

            def lesion_mask(axis, index):
                return ReportEngine._lesion_plane(lo, crop, shape_zyx, axis, index)

            # Napari (Z, Y, X) slicing:
            # Axial:    data[iz, :, :] -> shape (Y, X) -> Width=X, Height=Y -> aspect sx/sy
            # Coronal:  data[:, iy, :] -> shape (Z, X) -> Width=X, Height=Z -> aspect sx/sz
            # Sagittal: data[:, :, ix] -> shape (Z, Y) -> Width=Y, Height=Z -> aspect sy/sz
            slices = {
                "axial":    (ct_zyx[iz, :, :], pet_zyx[iz, :, :], lesion_mask(0, iz), sx, sy),
                "coronal":  (ct_zyx[:, iy, :], pet_zyx[:, iy, :], lesion_mask(1, iy), sx, sz),
                "sagittal": (ct_zyx[:, :, ix], pet_zyx[:, :, ix], lesion_mask(2, ix), sy, sz),
            }

            for plane, (ct_sl, pet_sl, mask_sl, step_w, step_h) in slices.items():
                ct_img = ReportEngine._render_slice(
                    ct_sl, mask_sl, ct_w, ct_l, ct_colormap, mask_opacity,
                )
                pet_img = ReportEngine._render_slice(
                    pet_sl, mask_sl, pet_w, pet_l, pet_colormap, mask_opacity,
                )

                # Apply aspect ratio scaling
                base_w, base_h = ct_img.size
                min_step = min(step_w, step_h)
                if min_step > 0:
                    new_w = int(round(base_w * (step_w / min_step)))
                    new_h = int(round(base_h * (step_h / min_step)))
                    if new_w != base_w or new_h != base_h:
                        resample_filter = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                        ct_img = ct_img.resize((new_w, new_h), resample_filter)
                        pet_img = pet_img.resize((new_w, new_h), resample_filter)

                ct_img.save(tumor_dir / f"{plane}_ct.png")
                pet_img.save(tumor_dir / f"{plane}_pet.png")

        # ── Dmax images ──
        if lesion_a and lesion_b:
            dmax_dir = images_dir / "dmax"
            dmax_dir.mkdir(exist_ok=True)

            dmax_mask_napari = np.zeros(shape_zyx, dtype=np.uint8)
            for les in (lesion_a, lesion_b):
                (z0, y0, x0), crop = crops[les["id"]]
                dz, dy, dx = crop.shape
                dmax_mask_napari[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx][crop] = 1

            p1 = app_coords(lesion_a["center_voxel"])
            p2 = app_coords(lesion_b["center_voxel"])

            ReportEngine.export_line_images(
                dmax_dir, ct_zyx, pet_zyx, dmax_mask_napari,
                p1, p2, (sx, sy, sz),
                (ct_w, ct_l), (pet_w, pet_l), ct_colormap, pet_colormap, mask_opacity,
            )
