import tkinter as tk

try:
    dialog = tk.Tk()
    dialog.title("Test")
    dialog.mainloop()
except Exception as e:
    print(f"Error: {e}")