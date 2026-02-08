import sys
import os
from pathlib import Path

# Add project root to path (so 'src' is importable)
root = Path(__file__).parent.parent
sys.path.append(str(root))

try:
    from PyQt6.QtWidgets import QApplication
    # Import from src package
    from src.gui.main_window import MainWindow
    from src.gui.components.viewer_widget import ViewerWidget
    from src.gui.components.control_panel import ControlPanel
    from src.gui.components.layout_manager import LayoutManager
    
    # Check imports of Core
    from src.core.session_manager import SessionManager
    from src.core.file_manager import FileManager
    
    print("Imports successful.")
    
    # Attempt instantiation (requires QApplication)
    app = QApplication([])
    
    # Test Components
    print("Testing Component Instantiation...")
    viewer = ViewerWidget()
    print("ViewerWidget: OK")
    
    control = ControlPanel()
    print("ControlPanel: OK")
    
    layout = LayoutManager()
    print("LayoutManager: OK")
    
    # Test Window
    window = MainWindow()
    print("MainWindow: OK")
    
    print("All GUI components verified successfully.")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
