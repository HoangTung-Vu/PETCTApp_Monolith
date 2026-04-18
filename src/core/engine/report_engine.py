"""Engine for computing PET/CT clinical report metrics (per-lesion)."""
import csv
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
from scipy.ndimage import label, find_objects

from ...utils.dimension_utils import get_voxel_volume_from_affine


class ReportEngine:
    """Computes clinical PET metrics on a per-lesion basis using connected
    component analysis.

    Per-lesion metrics:
        SUVmax  Maximum PET intensity within the lesion.
        SUVmean Mean PET intensity within the lesion.
        MTV     Metabolic Tumour Volume (mL) of the lesion.

    Global metrics:
        gTLG    Global Total Lesion Glycolysis = Σ(SUVmean_i × MTV_i).
    """

    @staticmethod
    def compute_report(
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
    ) -> dict:
        """Compute the per-lesion report dictionary.

        Uses connected component labeling to identify individual lesions in
        the binary mask.  Each lesion gets its own SUVmax, SUVmean, MTV, and
        3-D bounding box.

        Args:
            pet_image:  PET NIfTI image (intensity values = SUV or raw counts).
            mask_image: Binary segmentation mask NIfTI image (0/1).

        Returns:
            dict with keys:
                ``gTLG``    float, global Total Lesion Glycolysis.
                ``lesions`` list of dicts, each containing:
                    ``id``      int, 1-based lesion ID.
                    ``SUVmax``  float
                    ``SUVmean`` float
                    ``MTV``     float (mL)
                    ``bbox``    tuple (x_min, y_min, z_min, x_max, y_max, z_max)

        Raises:
            ValueError: If shapes mismatch or mask is empty.
        """
        pet_data = pet_image.get_fdata(dtype=np.float32)
        mask_data = np.asarray(mask_image.dataobj, dtype=np.uint8)

        if pet_data.shape != mask_data.shape:
            raise ValueError(
                f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}"
            )

        roi = mask_data > 0
        if not np.any(roi):
            raise ValueError("Mask is empty - no voxels to compute report on.")

        # Voxel volume derived from the affine determinant (source of truth)
        voxel_vol_mm3 = get_voxel_volume_from_affine(pet_image.affine)
        voxel_vol_ml = voxel_vol_mm3 / 1000.0

        # ── Connected component labeling ──
        labeled_array, num_features = label(roi)

        # find_objects returns tight bounding slices per component (O(N) total,
        # not O(N×L)) — avoids scanning the full array once per lesion.
        component_slices = find_objects(labeled_array)

        lesions = []
        g_tlg = 0.0

        for lesion_id, sl in enumerate(component_slices, start=1):
            if sl is None:
                continue

            # Work on the bounding-box subarray only
            sub_labeled = labeled_array[sl]
            sub_pet = pet_data[sl]
            component_mask = sub_labeled == lesion_id
            pet_vals = sub_pet[component_mask]

            suv_max = float(np.max(pet_vals))
            suv_mean = float(np.mean(pet_vals))
            n_voxels = len(pet_vals)
            mtv = n_voxels * voxel_vol_ml

            g_tlg += suv_mean * mtv

            # Bounding box from tight slices (find_objects guarantees tight fit)
            bbox = (
                sl[0].start, sl[1].start, sl[2].start,
                sl[0].stop - 1, sl[1].stop - 1, sl[2].stop - 1,
            )

            # Center voxel coordinates (NIfTI X, Y, Z)
            cx = (sl[0].start + sl[0].stop - 1) / 2.0
            cy = (sl[1].start + sl[1].stop - 1) / 2.0
            cz = (sl[2].start + sl[2].stop - 1) / 2.0

            lesions.append({
                "id": lesion_id,
                "SUVmax": round(suv_max, 2),
                "SUVmean": round(suv_mean, 2),
                "MTV": round(mtv, 2),
                "bbox": bbox,
                "center_voxel": (round(cx, 1), round(cy, 1), round(cz, 1)),
            })

        # Sort lesions by Z position (descending) so IDs go from head to feet.
        # In nibabel (X, Y, Z) order: bbox indices 2 and 5 are z_min and z_max.
        lesions.sort(key=lambda l: (l["bbox"][2] + l["bbox"][5]) / 2.0,
                     reverse=True)

        # Re-assign sequential IDs after sorting
        for idx, lesion in enumerate(lesions, start=1):
            lesion["id"] = idx

        # Compute physical coordinates using affine
        affine = pet_image.affine
        for lesion in lesions:
            cx, cy, cz = lesion["center_voxel"]
            phys = affine @ np.array([cx, cy, cz, 1.0])
            lesion["center_physical"] = (
                round(float(phys[0]), 2),
                round(float(phys[1]), 2),
                round(float(phys[2]), 2),
            )

        return {
            "gTLG": round(g_tlg, 2),
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

    @staticmethod
    def export_report(
        report_dir: Path,
        metrics: dict,
        ct_data: np.ndarray,
        pet_data: np.ndarray,
        mask_data: np.ndarray,
        affine: np.ndarray,
        ct_wl: tuple,
        pet_wl: tuple,
        ct_colormap: str = "gray",
        pet_colormap: str = "jet",
        mask_opacity: float = 0.5,
    ):
        """Export report CSV and per-tumor slice images.

        Wipes and recreates ``report_dir`` on every call.

        Args:
            report_dir: Target directory (e.g. storage/data/{sid}/report/).
            metrics: Dict returned by ``compute_report``.
            ct_data: CT volume in NIfTI (X, Y, Z) order, float32.
            pet_data: PET volume in NIfTI (X, Y, Z) order, float32.
            mask_data: Binary tumor mask in NIfTI (X, Y, Z) order, uint8.
            affine: 4x4 NIfTI affine matrix.
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

        from ...utils.nifti_utils import to_napari
        from ...utils.dimension_utils import get_spacing_from_affine
        
        ct_napari = to_napari(ct_data)
        pet_napari = to_napari(pet_data)
        mask_napari = to_napari(mask_data)

        sx, sy, sz = get_spacing_from_affine(affine)
        sx, sy, sz = abs(float(sx)), abs(float(sy)), abs(float(sz))
        
        shape_x, shape_y, shape_z = ct_data.shape

        lesions = metrics.get("lesions", [])
        g_tlg = metrics.get("gTLG", 0.0)

        # ── CSV ──
        csv_path = report_dir / "report_metabolic.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id",
                "voxel_x", "voxel_y", "voxel_z",
                "physical_x_mm", "physical_y_mm", "physical_z_mm",
                "SUVmax", "SUVmean", "MTV_mL", "TLG",
            ])
            for les in lesions:
                vx, vy, vz = les["center_voxel"]
                px, py, pz = les["center_physical"]
                tlg = round(les["SUVmean"] * les["MTV"], 2)
                
                # Convert NIfTI voxel coordinates to Napari (App GUI) coordinates
                app_z = shape_z - 1 - vz
                app_y = shape_y - 1 - vy
                app_x = vx
                
                writer.writerow([
                    les["id"],
                    int(round(app_x)), int(round(app_y)), int(round(app_z)),
                    px, py, pz,
                    les["SUVmax"], les["SUVmean"], les["MTV"], tlg,
                ])
            # Summary row
            writer.writerow([])
            writer.writerow(["gTLG", g_tlg])

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
                writer2.writerow([
                    "id", "voxel_x", "voxel_y", "voxel_z",
                    "physical_x_mm", "physical_y_mm", "physical_z_mm",
                    "SUVmax", "SUVmean", "MTV_mL", "TLG",
                ])
                for les in (lesion_a, lesion_b):
                    vx, vy, vz = les["center_voxel"]
                    px, py, pz = les["center_physical"]
                    tlg = round(les["SUVmean"] * les["MTV"], 2)
                    app_z = shape_z - 1 - vz
                    app_y = shape_y - 1 - vy
                    app_x = vx
                    writer2.writerow([
                        les["id"],
                        int(round(app_x)), int(round(app_y)), int(round(app_z)),
                        px, py, pz,
                        les["SUVmax"], les["SUVmean"], les["MTV"], tlg,
                    ])

        # ── Isolated Masks ──
        # Re-run connected components to map lesion IDs back to local mask labels
        from scipy.ndimage import label, find_objects
        labeled_array, _ = label(mask_data > 0)
        component_slices = find_objects(labeled_array)
        label_map = {}
        for index, sl in enumerate(component_slices, start=1):
            if sl is None: continue
            bbox = (sl[0].start, sl[1].start, sl[2].start, sl[0].stop - 1, sl[1].stop - 1, sl[2].stop - 1)
            for les in lesions:
                if les["bbox"] == bbox:
                    label_map[les["id"]] = index
                    break

        # ── Per-tumor images ──
        images_dir = report_dir / "images"
        images_dir.mkdir(exist_ok=True)

        ct_w, ct_l = ct_wl
        pet_w, pet_l = pet_wl

        for les in lesions:
            lid = les["id"]
            tumor_dir = images_dir / str(lid)
            tumor_dir.mkdir(exist_ok=True)

            cx, cy, cz = les["center_voxel"]
            
            # Use Napari coords for slicing
            app_z = shape_z - 1 - cz
            app_y = shape_y - 1 - cy
            app_x = cx
            
            iz, iy, ix = int(round(app_z)), int(round(app_y)), int(round(app_x))

            # Clamp to volume bounds (Napari shape)
            iz = max(0, min(iz, ct_napari.shape[0] - 1))
            iy = max(0, min(iy, ct_napari.shape[1] - 1))
            ix = max(0, min(ix, ct_napari.shape[2] - 1))

            # Isolate mask for this specific tumor
            mask_label = label_map.get(lid)
            if mask_label is not None:
                isolated_mask_data = (labeled_array == mask_label).astype(np.uint8)
                isolated_mask_napari = to_napari(isolated_mask_data)
            else:
                isolated_mask_napari = mask_napari  # Fallback

            # Napari (Z, Y, X) slicing:
            # Axial:    data[iz, :, :] -> shape (Y, X) -> Width=X, Height=Y -> aspect sx/sy
            # Coronal:  data[:, iy, :] -> shape (Z, X) -> Width=X, Height=Z -> aspect sx/sz
            # Sagittal: data[:, :, ix] -> shape (Z, Y) -> Width=Y, Height=Z -> aspect sy/sz
            slices = {
                "axial":    (ct_napari[iz, :, :], pet_napari[iz, :, :], isolated_mask_napari[iz, :, :], sx, sy),
                "coronal":  (ct_napari[:, iy, :], pet_napari[:, iy, :], isolated_mask_napari[:, iy, :], sx, sz),
                "sagittal": (ct_napari[:, :, ix], pet_napari[:, :, ix], isolated_mask_napari[:, :, ix], sy, sz),
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
            
            label_a = label_map.get(lesion_a["id"])
            label_b = label_map.get(lesion_b["id"])
            
            if label_a is not None and label_b is not None:
                dmax_mask_data = ((labeled_array == label_a) | (labeled_array == label_b)).astype(np.uint8)
            else:
                dmax_mask_data = mask_data
                
            dmax_mask_napari = to_napari(dmax_mask_data)
            
            # Map coordinates to Napari indices
            vza, vya, vxa = lesion_a["center_voxel"][2], lesion_a["center_voxel"][1], lesion_a["center_voxel"][0]
            vzb, vyb, vxb = lesion_b["center_voxel"][2], lesion_b["center_voxel"][1], lesion_b["center_voxel"][0]
            
            z1, y1, x1 = shape_z - 1 - vza, shape_y - 1 - vya, vxa
            z2, y2, x2 = shape_z - 1 - vzb, shape_y - 1 - vyb, vxb
            
            mid_z = max(0, min(int(round((z1 + z2) / 2)), ct_napari.shape[0] - 1))
            mid_y = max(0, min(int(round((y1 + y2) / 2)), ct_napari.shape[1] - 1))
            mid_x = max(0, min(int(round((x1 + x2) / 2)), ct_napari.shape[2] - 1))
            
            # Use MIP for PET and Mask, midpoint for CT
            pet_axial_mip = np.max(pet_napari, axis=0)
            mask_axial_mip = np.max(dmax_mask_napari, axis=0)
            
            pet_coronal_mip = np.max(pet_napari, axis=1)
            mask_coronal_mip = np.max(dmax_mask_napari, axis=1)
            
            pet_sagittal_mip = np.max(pet_napari, axis=2)
            mask_sagittal_mip = np.max(dmax_mask_napari, axis=2)
            
            dmax_slices = {
                "axial": (ct_napari[mid_z, :, :], pet_axial_mip, mask_axial_mip, sx, sy, (x1, y1), (x2, y2)),
                "coronal": (ct_napari[:, mid_y, :], pet_coronal_mip, mask_coronal_mip, sx, sz, (x1, z1), (x2, z2)),
                "sagittal": (ct_napari[:, :, mid_x], pet_sagittal_mip, mask_sagittal_mip, sy, sz, (y1, z1), (y2, z2)),
            }
            
            from PIL import ImageDraw
            for plane, (ct_sl, pet_sl, mask_sl, step_w, step_h, p1, p2) in dmax_slices.items():
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
                        
                # Scale coordinates to resized image
                p1_scaled = (p1[0] * (new_w / base_w), p1[1] * (new_h / base_h))
                p2_scaled = (p2[0] * (new_w / base_w), p2[1] * (new_h / base_h))
                
                # Draw a light red line connecting the tumors
                line_color = (255, 128, 128)
                img_draw_ct = ImageDraw.Draw(ct_img)
                img_draw_ct.line([p1_scaled, p2_scaled], fill=line_color, width=3)
                
                img_draw_pet = ImageDraw.Draw(pet_img)
                img_draw_pet.line([p1_scaled, p2_scaled], fill=line_color, width=3)
                
                ct_img.save(dmax_dir / f"{plane}_ct.png")
                pet_img.save(dmax_dir / f"{plane}_pet.png")
