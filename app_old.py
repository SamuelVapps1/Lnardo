import os
import tkinter as tk
from tkinter import ttk, messagebox
from dotenv import load_dotenv

def log(msg: str):
    log_box.configure(state="normal")
    log_box.insert("end", msg + "\n")
    log_box.see("end")
    log_box.configure(state="disabled")

def check_env():
    load_dotenv()
    api_key = os.getenv("LEONARDO_API_KEY")
    if not api_key:
        messagebox.showerror(
            "Missing API Key",
            "Missing LEONARDO_API_KEY.\n\nCreate a .env file (copy from .env.example) and paste your Leonardo API key."
        )
        return None
    return api_key

def on_test_api_key():
    api_key = check_env()
    if not api_key:
        return
    # len sanity check lokálne
    log("API key found in .env ✅ (Next: we will call Leonardo /me or a lightweight endpoint.)")

root = tk.Tk()
root.title("Laurapets Leonardo Tool")
root.geometry("780x520")

top = ttk.Frame(root, padding=12)
top.pack(fill="x")

ttk.Label(top, text="Laurapets – Leonardo Batch Generator", font=("Segoe UI", 14, "bold")).pack(anchor="w")
ttk.Label(top, text="Goal: generate 2 images per SKU (pack + pack) and save to disk.", font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))

btns = ttk.Frame(root, padding=12)
btns.pack(fill="x")

ttk.Button(btns, text="Test .env API Key", command=on_test_api_key).pack(side="left")

log_frame = ttk.Frame(root, padding=12)
log_frame.pack(fill="both", expand=True)

ttk.Label(log_frame, text="Log").pack(anchor="w")
log_box = tk.Text(log_frame, height=20, wrap="word", state="disabled")
log_box.pack(fill="both", expand=True)

root.mainloop()
