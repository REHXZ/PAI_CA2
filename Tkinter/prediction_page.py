import tkinter as tk
from tkinter import ttk

class PredictionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Prediction Page UI
        tk.Label(self, text="Customer ID").grid(row=0, column=0, padx=10, pady=10)
        self.customer_id_entry = tk.Entry(self)
        self.customer_id_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self, text="Order ID").grid(row=1, column=0, padx=10, pady=10)
        self.order_id_entry = tk.Entry(self)
        self.order_id_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self, text="Country Code").grid(row=2, column=0, padx=10, pady=10)
        self.country_code_combobox = ttk.Combobox(self, values=["SG", "MY", "US", "CN"])
        self.country_code_combobox.grid(row=2, column=1, padx=10, pady=10)

        predict_button = tk.Button(self, text="Predict", command=self.predict)
        predict_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.grid(row=4, column=0, columnspan=2)

        analysis_button = tk.Button(self, text="View Analysis", command=lambda: controller.show_frame("AnalysisPage"))
        analysis_button.grid(row=5, column=0, columnspan=2, pady=10)

    def predict(self):
        # Placeholder for prediction logic
        customer_id = self.customer_id_entry.get()
        order_id = self.order_id_entry.get()
        country_code = self.country_code_combobox.get()

        # Simulate prediction (replace with actual model prediction)
        if customer_id and order_id and country_code:
            self.result_label.config(text=f"Prediction: Not Fraud (Sample Result)")
        else:
            self.result_label.config(text="Please fill all fields")