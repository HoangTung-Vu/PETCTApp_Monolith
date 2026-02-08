import sys
import os
from PyQt6.QtWidgets import QApplication
from .gui.main_window import MainWindow
from .database.db import init_db

def main():
    # Initialize Database
    init_db()

    # Initialize Application
    app = QApplication(sys.argv)
    
    # Napari uses its own Qt app if one doesn't exist, but we created one.
    # We might need to handle event loops carefully if using napari.run(), 
    # but with embedding, we control the loop via app.exec().
    
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
