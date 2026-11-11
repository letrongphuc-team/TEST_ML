import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

from SalesDatabaseProject.ui.LoginMainWindowEx import LoginMainWindowEx

app = QApplication(sys.argv)
main_window = QMainWindow()
myui = LoginMainWindowEx()
myui.setupUi(main_window)
myui.ShowWindow()

sys.exit(app.exec())