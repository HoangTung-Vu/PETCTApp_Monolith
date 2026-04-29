import napari
viewer = napari.Viewer(show=False)
print("Callbacks:", viewer.mouse_double_click_callbacks)
