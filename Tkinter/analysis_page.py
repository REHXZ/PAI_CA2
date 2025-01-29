import tkinter as tk
from tkinter import messagebox

class AnalysisPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Analysis Page UI
        tk.Label(self, text="Feedback and Analysis").pack(pady=10)

        self.feedback_text = tk.Text(self, height=10, width=50)
        self.feedback_text.pack(pady=10)

        submit_button = tk.Button(self, text="Submit Feedback", command=self.submit_feedback)
        submit_button.pack(pady=10)

        back_button = tk.Button(self, text="Back to Prediction", command=lambda: controller.show_frame("PredictionPage"))
        back_button.pack(pady=10)

    def submit_feedback(self):
        feedback = self.feedback_text.get("1.0", tk.END).strip()
        if feedback:
            messagebox.showinfo("Feedback Submitted", "Thank you for your feedback!")
            self.feedback_text.delete("1.0", tk.END)
        else:
            messagebox.showwarning("Empty Feedback", "Please enter some feedback before submitting.")