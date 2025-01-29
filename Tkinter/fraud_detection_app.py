import tkinter as tk
from login_page import LoginPage
from prediction_page import PredictionPage
from analysis_page import AnalysisPage

class FraudDetectionApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Fraud Detection System")
        self.geometry("500x400")

        # Container to hold all frames
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Dictionary to hold all frames
        self.frames = {}

        for F in (LoginPage, PredictionPage, AnalysisPage):
            frame = F(container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Show the login page first
        self.show_frame("LoginPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()