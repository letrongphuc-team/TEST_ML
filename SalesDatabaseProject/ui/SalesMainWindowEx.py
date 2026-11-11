from PyQt6.QtWidgets import QTableWidgetItem, QMainWindow

from SalesDatabaseProject.ui.ChartVisualizationMainWindowEx import ChartVisualizationMainWindowEx
from SalesDatabaseProject.ui.SalesMainWindow import Ui_MainWindow
from connectors.connector import Connector


class SalesMainWindowEx(Ui_MainWindow):
    def __init__(self):
        self.conn = Connector()
        self.conn.connect()
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow
        self.setupSignalAndSlot()
    def showWindow(self):
        self.MainWindow.show()
    def showDataIntoTableWidget(self,table,df):
        table.setRowCount(0)
        table.setColumnCount(len(df.columns))
        for i in range(len(df.columns)):
            columnHeader=df.columns[i]
            table.setHorizontalHeaderItem(i, QTableWidgetItem(columnHeader))
        row=0
        for item in df.iloc:
            arr=item.values.tolist()
            table.insertRow(row)
            j=0
            for data in arr:
                table.setItem(row,j,QTableWidgetItem(str(data)))
                j=j+1
            row=row+1

    def setupSignalAndSlot(self):
        self.pushButtonCustomer.clicked.connect(self.showCustomer)
        self.pushButtonSpendScore.clicked.connect(self.showCustomerSpendScore)
        self.pushButtonStatistic.clicked.connect(self.showVisualizationChart)
    def showCustomer(self):
        sql = "select * from customer"
        df = self.conn.queryDataset(sql)
        self.showDataIntoTableWidget(self.tableWidgetData,df)
    def showCustomerSpendScore(self):
        sql = "select * from customer_spend_score"
        df = self.conn.queryDataset(sql)
        self.showDataIntoTableWidget(self.tableWidgetData, df)
    def showVisualizationChart(self):
        self.myui = ChartVisualizationMainWindowEx()
        self.myui.setupUi(QMainWindow())
        self.myui.showWindow()

