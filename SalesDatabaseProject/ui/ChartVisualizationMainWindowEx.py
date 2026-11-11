from SalesDatabaseProject.ui.ChartVisualizationMainWindow import Ui_MainWindow
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns
from connectors.connector import Connector


class ChartVisualizationMainWindowEx(Ui_MainWindow):

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow
        self.setupSignalAndSlot()
        self.setupPlot()

    def showWindow(self):
        self.MainWindow.show()

    def setupSignalAndSlot(self):
        self.pushButtonCustomer.clicked.connect(
            self.showPiechartForCustomer
        )

        self.pushButtonCustomerSpendScore.clicked.connect(
            self.showLineChartForCustomer
        )

    def connectDatabase(self):
        self.connector = Connector()
        self.connector.connect()

    def setupPlot(self):
        self.figure=plt.figure()
        self.canvas=FigureCanvas(self.figure)
        self.toolbar=NavigationToolbar(self.canvas,self.MainWindow)

        self.verticalLayoutPlot.addWidget(self.toolbar)
        self.verticalLayoutPlot.addWidget(self.canvas)

    def showPiechartForCustomer(self):
        conn = Connector()
        conn.connect()

        sql = "select * from customer"

        df = conn.queryDataset(sql)

        dfGender = (
            df["Gender"]
            .value_counts()
            .rename_axis("Gender")
            .reset_index(name="Count")
        )

        print(dfGender)

        columnlabel = "Gender"
        columnStatistic = "Count"
        title = "Gender Distribution"

        print(dfGender[columnStatistic])
        print(dfGender[columnlabel])

        legend = True

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        ax.pie(
            dfGender[columnStatistic],
            labels=dfGender[columnlabel],
            autopct='%1.2f%%'
        )

        if legend:
            ax.legend(
                dfGender[columnlabel],
                loc='lower right'
            )

        ax.set_title(title)

        self.canvas.draw()

    def showLineChartForCustomer(self):
        self.connectDatabase()

        sql = "select distinct customer_spend_score.*,customer.Age " \
              "from customer, customer_spend_score " \
              "where customer.CustomerID=customer_spend_score.CustomerID"

        df = self.connector.queryDataset(sql)

        fromAge = int(self.lineEditFromAge.text())
        toAge = int(self.lineEditToAge.text())

        dfAges = df[
            (df.Age >= fromAge) &
            (df.Age <= toAge)
        ]

        dfAges.sort_values(
            by=['Age'],
            ascending=True,
            inplace=True
        )

        columnLabel = "Age"
        columnStatistic = "Spending_Score"

        title = "Age Distribution %s~%s" % (
            fromAge,
            toAge
        )

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        ax.ticklabel_format(
            useOffset=False,
            style="plain"
        )

        ax.grid()

        sns.lineplot(
            data=dfAges,
            x=columnLabel,
            y=columnStatistic,
            marker='o',
            color='orange',
            ax=ax
        )

        ax.set_xlabel(columnLabel)

        ax.set_ylabel(columnStatistic)

        ax.set_title(title)

        self.canvas.draw()