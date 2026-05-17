# Paint Tool Optimization Plan

## 1. Vấn đề hiện tại
- Hiện tại, mỗi khi người dùng vẽ xong 1 nét (ngừng chuột 300ms), hàm `_on_auto_sync` sẽ gọi `from_napari` để convert mảng `(Z, Y, X)` thành `(X, Y, Z)` nhằm lưu vào `session_manager`. 
- Quá trình convert này tạo ra mảng Numpy 3D mới bằng ThreadPool trên Main Thread, gây ra hiện tượng lag/khựng UI.

## 2. Ý tưởng tối ưu (Sử dụng "Confirm ROI")
Thay vì auto-sync dữ liệu nặng mỗi 300ms, ta sẽ:
1. **Tắt deep copy trong `_on_auto_sync`**: Cho phép vẽ thoải mái mà không gọi `from_napari`. 
2. **Tận dụng nút "Confirm ROI"**: Hàm `_on_confirm_roi` vốn đã có lệnh `_sync_roi_from_viewer()`. Người dùng vẽ xong sẽ bấm nút này để chủ động chạy quá trình convert `from_napari`.
3. **Đồng bộ Tumor thủ công trước khi Save**: Với Tumor mask, ta sẽ gom việc convert lại vào lúc người dùng bấm "Confirm & Save".

---

## 3. Các bước thực hiện chi tiết

### Bước 1: Thêm hàm lấy dữ liệu ZYX nhanh vào `layout_manager.py`
Để nút "Refine" vẫn có thể sáng lên ngay khi có nét vẽ (dựa vào hàm `_is_roi_dirty`), ta cần lấy trực tiếp mảng Napari đang có trên Viewer thay vì gọi `from_napari`.
- **Thêm hàm**: `get_active_mask_data_zyx(self, layer_type: str)` vào `layout_manager.py`. Hàm này chỉ trả về `viewer.layers[layer_type].data`.

### Bước 2: Tối ưu `_is_roi_dirty` trong `refinement_handler.py`
Sửa hàm `_is_roi_dirty` để check dữ liệu từ hàm `get_active_mask_data_zyx` vừa tạo:
```python
def _is_roi_dirty(self) -> bool:
    # Lấy trực tiếp Napari array (Z, Y, X) để check cực nhanh (O(1))
    zyx = self.layout_manager.get_active_mask_data_zyx("roi")
    if zyx is not None:
        return bool(zyx.any())
    
    # Fallback cho session_manager
    roi_data = self.session_manager.get_roi_mask_data()
    return bool(roi_data is not None and roi_data.any())
```

### Bước 3: Loại bỏ deep copy khỏi `_on_auto_sync`
Sửa hàm `_on_auto_sync` trong `refinement_handler.py` để KHÔNG gọi `get_active_mask_data()` nữa. Nó chỉ làm nhiệm vụ cập nhật UI:
```python
def _on_auto_sync(self, layer_type: str):
    # Loại bỏ from_napari (get_active_mask_data)
    
    if layer_type == "roi" and not getattr(self, '_preview_active', False):
        self._update_refine_button_states()
        print("[AutoSync] ROI painted. Đợi người dùng bấm Confirm ROI.")
        
    elif layer_type == "tumor":
        self.session_manager.clear_lesion_data()
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)
        print("[AutoSync] Tumor painted. Xóa Lesion IDs cache.")
```

### Bước 4: Đảm bảo Tumor mask được convert trước khi Save
Bởi vì `_on_auto_sync` không còn lưu Tumor mask vào `session_manager` nữa, ta phải đảm bảo nó được convert trước khi lưu file.
- **Thêm hàm**: `_sync_tumor_from_viewer(self)` vào `refinement_handler.py`.
- **Gọi hàm này**: Đặt vào đầu hàm `_on_confirm_and_save()` để lấy các nét vẽ manual (nếu có) trước khi ném cho `MergeSaveWorker` lưu xuống đĩa. Cần gọi ở `_on_refinement_tab_changed` nếu có leave tab.

## 4. Kết quả mong đợi
- **Khi vẽ Paint**: Mượt hoàn toàn, không có độ trễ do bỏ qua `from_napari`.
- **Đồng bộ Viewer**: Các viewer vẫn hiển thị nét vẽ của nhau tức thời nhờ cơ chế `layer.refresh()` đã có trong `mask_sync.py`.
- **Chuyển đổi Napari**: Quá trình nặng chỉ xảy ra đúng 1 lần khi bấm nút **"Confirm ROI"** hoặc **"Confirm & Save"**.

*(Phần Lesion ID sẽ được giữ nguyên theo yêu cầu của bạn, không can thiệp).*
