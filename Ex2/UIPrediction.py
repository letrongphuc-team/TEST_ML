from tkinter import *
from tkinter import messagebox, ttk
from tkinter.font import Font
from tkinter import filedialog as fd

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from DataSetViewer import DataSetViewer
from FileUtil import FileUtil

FEATURES = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms', 'Area Population']


class UIPrediction:
    def __init__(self):
        self.lm = None

    # ------------------------------------------------------------------ UI
    def create_ui(self):
        self.root = Tk()
        self.root.title("House Pricing Prediction - Faculty of Information Systems")
        self.root.geometry("1500x850")
        main_panel = PanedWindow(self.root)
        main_panel["bg"] = "yellow"
        main_panel.pack(fill=BOTH, expand=True)

        # --- tiêu đề
        top_panel = PanedWindow(main_panel, height=80)
        top_panel["bg"] = "blue"
        main_panel.add(top_panel)
        top_panel.pack(fill=X, side=TOP, expand=False)
        font = Font(family="tahoma", size=18)
        title_label = Label(top_panel, text='House Pricing Prediction', font=font)
        title_label["bg"] = "yellow"
        top_panel.add(title_label)

        center_panel = PanedWindow(main_panel)
        main_panel.add(center_panel)
        center_panel["bg"] = "pink"
        center_panel.pack(fill=BOTH, expand=True)

        # --- chọn dataset (nút 1, 2)
        choose_dataset_panel = PanedWindow(center_panel, height=30)
        center_panel.add(choose_dataset_panel)
        choose_dataset_panel["bg"] = "orange"
        choose_dataset_panel.pack(fill=X)

        dataset_label = Label(choose_dataset_panel, text="Select Dataset:")
        self.selectedFileName = StringVar()
        self.selectedFileName.set("dataset/USA_Housing.csv")
        self.choose_dataset_entry = Entry(choose_dataset_panel, textvariable=self.selectedFileName)
        self.choose_dataset_button = Button(choose_dataset_panel, text="1.Pick Dataset",
                                            width=10, command=self.do_pick_data)
        self.view_dataset_button = Button(choose_dataset_panel, text="2.View Dataset",
                                          width=20, command=self.do_view_dataset)
        choose_dataset_panel.add(dataset_label)
        choose_dataset_panel.add(self.choose_dataset_entry)
        choose_dataset_panel.add(self.choose_dataset_button)
        choose_dataset_panel.add(self.view_dataset_button)
        self.view_dataset_button.pack(side=RIGHT, expand=False)
        self.choose_dataset_button.pack(side=RIGHT, expand=False)

        # --- training rate (nút 3, 4)
        training_rate_panel = PanedWindow(center_panel, height=30)
        center_panel.add(training_rate_panel)
        training_rate_panel.pack(fill=X)
        training_rate_label = Label(training_rate_panel, text="Training Rate:")
        self.training_rate = IntVar()
        self.training_rate.set(80)
        self.training_rate_entry = Entry(training_rate_panel, textvariable=self.training_rate, width=20)
        training_rate_panel.add(training_rate_label)
        training_rate_panel.add(self.training_rate_entry)
        percent_label = Label(training_rate_panel, text="%", width=20, anchor="w", justify=LEFT)
        percent_label.pack(side=RIGHT, expand=False, fill=X)
        training_rate_panel.add(percent_label)
        self.train_model_button = Button(training_rate_panel, text="3.Train Model",
                                         width=20, command=self.do_train)
        training_rate_panel.add(self.train_model_button)
        self.evaluate_model_button = Button(training_rate_panel, text="4.Evaluate Model",
                                            width=20, command=self.do_evaluation)
        training_rate_panel.add(self.evaluate_model_button)
        self.status = StringVar()
        self.train_model_result_label = Label(training_rate_panel, textvariable=self.status)
        training_rate_panel.add(self.train_model_result_label)

        # --- bảng kết quả TEST + Coefficient
        evaluate_panel = PanedWindow(center_panel, height=400)
        evaluate_panel["bg"] = "cyan"
        center_panel.add(evaluate_panel)
        evaluate_panel.pack(fill=X)

        table_evaluate_panel = PanedWindow(evaluate_panel, height=400)
        evaluate_panel.add(table_evaluate_panel)

        columns = FEATURES + ['Original Price', 'Prediction Price']
        self.tree = ttk.Treeview(table_evaluate_panel, columns=columns, show="headings")
        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=CENTER, stretch=NO, width=140)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar = ttk.Scrollbar(table_evaluate_panel, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)

        coefficient_panel = PanedWindow(evaluate_panel)
        coefficient_panel["bg"] = "pink"
        coefficient_panel.pack(side=RIGHT, fill=X, expand=False)
        evaluate_panel.add(coefficient_panel)

        coefficient_detail_label = Label(coefficient_panel, text="Coefficient:")
        coefficient_panel.add(coefficient_detail_label)
        coefficient_detail_label.pack(side=TOP, fill=X, expand=False)

        coefficient_detail_panel = PanedWindow(coefficient_panel)
        coefficient_panel.add(coefficient_detail_panel)
        coefficient_detail_panel.pack(side=TOP, expand=False, fill=X)
        self.coefficient_detail_text = Text(coefficient_detail_panel, height=12, width=50)
        scroll = Scrollbar(coefficient_detail_panel)
        self.coefficient_detail_text.configure(yscrollcommand=scroll.set)
        self.coefficient_detail_text.pack(side=LEFT, expand=False, fill=X)
        scroll.config(command=self.coefficient_detail_text.yview)
        scroll.pack(side=RIGHT, fill=Y, expand=True)

        # --- MAE / MSE / RMSE + nút 5 Save Model
        metric_panel = PanedWindow(coefficient_panel, height=30)
        coefficient_panel.add(metric_panel)
        metric_panel.pack(side=TOP, fill=BOTH, expand=True)

        self.mae_value = DoubleVar()
        Label(metric_panel, text="Mean Absolute Error(MAE):").grid(row=0, column=0)
        Entry(metric_panel, width=20, textvariable=self.mae_value).grid(row=0, column=1)
        self.mse_value = DoubleVar()
        Label(metric_panel, text="Mean Square Error(MSE):").grid(row=1, column=0)
        Entry(metric_panel, width=20, textvariable=self.mse_value).grid(row=1, column=1)
        self.rmse_value = DoubleVar()
        Label(metric_panel, text="Root Mean Square Error(RMSE):").grid(row=2, column=0)
        Entry(metric_panel, width=20, textvariable=self.rmse_value).grid(row=2, column=1)
        Button(metric_panel, text="5. Save Model", width=20,
               command=self.do_save_model).grid(row=3, column=1)

        # --- nút 6 Load Model
        loadmodel_panel = PanedWindow(center_panel, height=20)
        loadmodel_panel["bg"] = "yellow"
        loadmodel_panel.pack(fill=BOTH, side=TOP)
        Button(loadmodel_panel, text="6. Load Model",
               command=self.do_load_model).grid(row=0, column=0)

        # --- dự báo đơn lẻ (nút 7)
        p = PanedWindow(center_panel)
        p.pack(fill=BOTH, side=TOP, expand=True)
        labels = ["Avg. Area Income:", "Avg. Area House Age:", "Avg. Area Number of Rooms:",
                  "Avg. Area Number of Bedrooms:", "Area Population:"]
        self.input_vars = []
        for r, text in enumerate(labels):
            Label(p, text=text).grid(row=r, column=0)
            v = DoubleVar()
            self.input_vars.append(v)
            Entry(p, width=40, textvariable=v).grid(row=r, column=1)
        Button(p, text="7. Prediction House Pricing",
               command=self.do_prediction).grid(row=5, column=1)
        Label(p, text="Prediction Price:").grid(row=6, column=0)
        self.prediction_price_value = DoubleVar()
        Entry(p, width=40, textvariable=self.prediction_price_value).grid(row=6, column=1)

        # --- dự báo theo danh sách Excel (nâng cao)
        Button(p, text="8. Predict from Excel/CSV list",
               command=self.do_predict_from_file).grid(row=7, column=1)

        designedby_panel = PanedWindow(main_panel, height=20)
        designedby_panel["bg"] = "cyan"
        designedby_panel.pack(fill=BOTH, side=BOTTOM)
        designedby_label = Label(designedby_panel, text="Designed by: Tran Duy Thanh")
        designedby_label["bg"] = "cyan"
        designedby_label.pack(side=LEFT)

    def show_ui(self):
        self.root.mainloop()

    # -------------------------------------------------------------- events
    def do_pick_data(self):
        filetypes = (("Dataset CSV", "*.csv"), ("All Files", "*.*"))
        s = fd.askopenfilename(title="Choose dataset", initialdir="/", filetypes=filetypes)
        if s:                                # người dùng bấm Cancel thì s == ""
            self.selectedFileName.set(s)

    def do_view_dataset(self):
        viewer = DataSetViewer()
        viewer.create_ui()
        viewer.show_data_listview(self.selectedFileName.get())
        viewer.show_ui()

    def do_train(self):
        try:
            ratio = self.training_rate.get() / 100
            self.df = pd.read_csv(self.selectedFileName.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.X = self.df[FEATURES]
        self.y = self.df['Price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=1 - ratio, random_state=101)
        self.lm = LinearRegression()
        self.lm.fit(self.X_train, self.y_train)
        self.status.set("Trained is finished")
        messagebox.showinfo("info", "Trained is finished")

    def do_evaluation(self):
        if self.lm is None or not hasattr(self, "X_test"):
            messagebox.showwarning("Warning", "Hãy train model (bước 3) trước!")
            return
        self.tree.delete(*self.tree.get_children())
        self.coefficient_detail_text.delete("1.0", END)

        self.coeff_df = pd.DataFrame(self.lm.coef_, self.X.columns, columns=['Coefficient'])
        self.coefficient_detail_text.insert(END, self.coeff_df)

        predictions = self.lm.predict(self.X_test)
        y_test_array = np.asarray(self.y_test)
        for i in range(len(self.X_test)):
            values = list(self.X_test.iloc[i]) + [y_test_array[i], predictions[i]]
            self.tree.insert('', END, values=values)

        self.mae_value.set(metrics.mean_absolute_error(self.y_test, predictions))
        self.mse_value.set(metrics.mean_squared_error(self.y_test, predictions))
        self.rmse_value.set(np.sqrt(metrics.mean_squared_error(self.y_test, predictions)))
        self.status.set("Evaluation is finished")
        messagebox.showinfo("info", "Evaluation is finished")

    def do_save_model(self):
        if self.lm is None:
            messagebox.showwarning("Warning", "Chưa có model để lưu!")
            return
        if FileUtil.savemodel(self.lm, "housingmodel.zip"):
            messagebox.showinfo("info", "exported model to disk successful!")

    def do_load_model(self):
        model = FileUtil.loadmodel("housingmodel.zip")
        if model is None:
            messagebox.showerror("Error", "Không tìm thấy / không đọc được housingmodel.zip")
            return
        self.lm = model
        messagebox.showinfo("info", "loading model from disk successful!")

    def do_prediction(self):
        if self.lm is None:
            messagebox.showwarning("Warning", "Hãy train hoặc load model trước!")
            return
        try:
            row = [[v.get() for v in self.input_vars]]
        except TclError:
            messagebox.showerror("Error", "Vui lòng nhập số hợp lệ ở tất cả các ô")
            return
        result = self.lm.predict(pd.DataFrame(row, columns=FEATURES))
        self.prediction_price_value.set(result[0])

    def do_predict_from_file(self):
        """Dự báo hàng loạt: đọc file Excel/CSV có đủ 5 cột đặc trưng, xuất file kết quả."""
        if self.lm is None:
            messagebox.showwarning("Warning", "Hãy train hoặc load model trước!")
            return
        path = fd.askopenfilename(title="Choose Excel/CSV list",
                                  filetypes=(("Excel", "*.xlsx"), ("CSV", "*.csv")))
        if not path:
            return
        try:
            data = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
            data["Prediction Price"] = self.lm.predict(data[FEATURES])
            out = path.rsplit(".", 1)[0] + "_predicted.xlsx"
            data.to_excel(out, index=False)
            messagebox.showinfo("info", f"Đã xuất kết quả: {out}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
