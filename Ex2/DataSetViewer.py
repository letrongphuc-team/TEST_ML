from tkinter import *
from tkinter import ttk
import pandas as pd


class DataSetViewer:
    """Cửa sổ xem dữ liệu gốc (Bước 2: View Dataset)."""

    def __init__(self):
        pass

    def create_ui(self):
        self.root = Toplevel()          # Toplevel để không xung đột với cửa sổ chính
        self.root.title("Dataset viewer - House Pricing Prediction")
        self.root.geometry("800x600")
        main_panel = PanedWindow(self.root)
        main_panel["bg"] = "yellow"
        main_panel.pack(fill=BOTH, expand=True)

        columns = ('Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                   'Avg. Area Number of Bedrooms', 'Area Population', 'Price')
        self.tree = ttk.Treeview(main_panel, columns=columns, show="headings")
        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=CENTER, width=120)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(main_panel, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)

    def show_ui(self):
        # Toplevel chạy chung mainloop của cửa sổ chính, không gọi mainloop lần nữa
        self.root.focus_set()

    def show_data_listview(self, fileName):
        df = pd.read_csv(fileName)
        for i in range(len(df)):
            values = df.iloc[i, :6].tolist()
            self.tree.insert('', END, values=values)
