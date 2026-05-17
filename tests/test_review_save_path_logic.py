"""
REVIEW TEST — `SessionManager.save_session` path-fallback logic
==============================================================

Hàm `save_session` (src/core/session_manager.py:175):

    if session.tumor_seg_path and Path(session.tumor_seg_path).parent.exists():
        seg_path = Path(session.tumor_seg_path)
    else:
        ct_path = Path(session.ct_path) if session.ct_path else None
        if ct_path and ct_path.exists():
            seg_path = FileManager.get_segmentation_path(ct_path)
        else:
            seg_path = FileManager.get_file_path(self.current_session_id, "tumor_seg")

PROBLEM:
  - `parent.exists()` returns True for ANY existing folder, kể cả khi file
    tumor_seg_path đã bị xóa, hoặc thư mục không còn quyền ghi.
  - DB còn nhớ path cũ → nếu user MOVE file CT sang folder mới và load file CT
    mới qua dialog: `update_current_session` chỉ update `ct_path`,
    `tumor_seg_path` vẫn trỏ đến file gốc. Khi save, parent vẫn exist →
    save vẫn đi vào folder cũ, KHÔNG nằm cạnh CT mới như mong muốn.

Test này verify behavior trên đường dẫn giả lập (không dùng SQLAlchemy).
"""

import tempfile
from pathlib import Path
import shutil


def determine_save_path(tumor_seg_path: str | None, ct_path: str | None) -> str:
    """Replica logic from SessionManager.save_session, simplified."""
    if tumor_seg_path and Path(tumor_seg_path).parent.exists():
        return tumor_seg_path
    if ct_path and Path(ct_path).exists():
        # FileManager.get_segmentation_path semantics
        p = Path(ct_path)
        name = p.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                stem = name[: -len(ext)]
                break
        else:
            stem = p.stem
        return str(p.parent / f"{stem}_Segmentation.nii.gz")
    return "legacy"


def main():
    print("=== Save-path logic — bug demo ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Setup: user originally had CT in /old, with seg saved next to it.
        old_dir = tmp / "old"
        old_dir.mkdir()
        old_ct = old_dir / "patient1.nii.gz"
        old_ct.write_bytes(b"")
        old_seg = old_dir / "patient1_Segmentation.nii.gz"
        old_seg.write_bytes(b"")

        # Now: user loads a NEW CT from /new via dialog → DB updates ct_path,
        # but tumor_seg_path còn ghi /old/patient1_Segmentation.nii.gz.
        new_dir = tmp / "new"
        new_dir.mkdir()
        new_ct = new_dir / "patient2.nii.gz"
        new_ct.write_bytes(b"")

        save_path = determine_save_path(str(old_seg), str(new_ct))
        print(f"Old CT:   {old_ct}")
        print(f"Old seg:  {old_seg}")
        print(f"New CT:   {new_ct}")
        print(f"→ save_session sẽ ghi vào: {save_path}")
        expected = new_dir / "patient2_Segmentation.nii.gz"
        if Path(save_path) == old_seg:
            print(f"\n⚠️  BUG: seg ghi vào folder CŨ ({old_dir.name}/), không phải cạnh CT mới.")
            print(f"    User update CT qua dialog nhưng segmentation lưu sai chỗ.")
            print(f"    Expected: {expected}")
        else:
            print(f"\n✓ OK: seg ghi cạnh CT mới.")

        # Phần 2: thư mục seg vẫn còn (parent.exists()) nhưng file gốc đã bị xóa.
        old_seg.unlink()
        save_path2 = determine_save_path(str(old_seg), str(new_ct))
        print(f"\n--- Scenario 2: file seg cũ đã xóa, folder vẫn còn ---")
        print(f"→ save_session sẽ ghi vào: {save_path2}")
        if Path(save_path2) == old_seg:
            print("⚠️  BUG: tạo lại file seg ở folder /old/ thay vì cạnh CT mới /new/.")
        else:
            print("✓ OK")


if __name__ == "__main__":
    main()
