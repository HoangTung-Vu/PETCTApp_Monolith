"""
REVIEW TEST — MRO bug: `LayoutManager._get_visible_viewers` override mixin
=========================================================================

Mục đích kiểm tra: `MaskSyncMixin._get_visible_viewers` được thiết kế để TRẢ VỀ
3D viewer khi stack đang ở 3D, hoặc 2D pool khi đang 2D. Nhưng `LayoutManager`
ĐỊNH NGHĨA LẠI `_get_visible_viewers` (line 815, layout_manager.py) bằng
alias đến `_get_all_loaded_viewers()` — TRẢ VỀ CẢ 2D POOL + 3D VIEWER (khi
3D đã loaded). MRO của Python ưu tiên class body trước mixin → mixin's version
KHÔNG BAO GIỜ chạy.

Hệ quả:
  - khi user đang ở 3D mode mà paint mask: mask events fire trên CẢ 9 viewer 2D
    chưa hiển thị + 3D viewer → `_do_visual_refresh` gọi `layer.refresh()` trên
    9 viewer không cần thiết.
  - khi 3D đã loaded rồi quay lại 2D: tương tự, 3D viewer cũng được include.

Test này dùng minimal mock để chứng minh MRO bug.
"""

class MaskSyncMixin:
    def _get_visible_viewers(self):
        return "mixin_version"


class LayoutManager(MaskSyncMixin):
    def _get_visible_viewers(self):
        # đây là override trong class body
        return "class_version"


def main():
    lm = LayoutManager()
    result = lm._get_visible_viewers()
    print(f"MRO chain: {[c.__name__ for c in LayoutManager.__mro__]}")
    print(f"_get_visible_viewers() returns: {result!r}")
    assert result == "class_version", "Class body always overrides mixin"
    print("\n✅ Bug confirmed: LayoutManager.body._get_visible_viewers OVERRIDES")
    print("   MaskSyncMixin._get_visible_viewers.")
    print("   The mixin version (which distinguishes 2D-vs-3D stack widget)")
    print("   never executes. _connect_mask_events / _do_visual_refresh therefore")
    print("   iterate ALL 9 pool viewers AND the 3D viewer (when loaded) — not")
    print("   just the currently-rendered ones.")
    print("\nFix: rename one of them, e.g. mixin uses `_get_active_viewers` and")
    print("     LayoutManager._get_visible_viewers calls into it, or remove the")
    print("     duplicate override.")


if __name__ == "__main__":
    main()
