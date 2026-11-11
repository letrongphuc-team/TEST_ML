from SalesDatabaseProject.ui.LoginMainWindow import Ui_MainWindow

from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox


class LoginMainWindowEx(Ui_MainWindow):
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        self.setupSignalAndSlot()
    def ShowWindow(self):
        self.MainWindow.show()

    def setupSignalAndSlot(self):
        self.pushButtonLogin.clicked.connect(self.process_login)
    def process_login(self):
        from SalesDatabaseProject.dal.user_dao import UserDao

        uid = self.lineEditUserName.text()
        pwd = self.lineEditPassword.text()
        us = UserDao.login(uid, pwd)
        self.msgBox = QMessageBox()
        if us is None:
            self.msgBox.setText("Login Failed")
        else:
            self.msgBox.setText("Login success")
        self.msgBox.exec()