import tkinter as tk
from tkinter import Canvas, Label, Button
import pyautogui
import cv2
import numpy as np
import time
from PIL import Image, ImageTk

# Global variables
running = True
game_region = None

# Function to exit the program
def exit_program():
    global running
    running = False

def show_initial_dialog():
    try:
        dialog = tk.Tk()
    except Exception as e:
        print(f"Failed to initialize Tkinter: {e}")
        exit_program()
        return
    dialog.attributes('-topmost', True)
    dialog.title("Ready to Select Game Area")
    screen_width = dialog.winfo_screenwidth()
    dialog_width = 400
    dialog_height = 100
    dialog.geometry(f"{dialog_width}x{dialog_height}+{(screen_width - dialog_width) // 2}+50")  # 50 pixels from the top

    label = Label(dialog, text="Click Confirm button when you are ready to select the game area")
    label.pack(pady=10)
    
    def on_confirm():
        dialog.destroy()
        dialog.update()
    
    def on_cancel():
        exit_program()
        dialog.destroy()
    
    confirm_button = Button(dialog, text="Confirm", command=on_confirm)
    confirm_button.pack(side="left", padx=20)
    cancel_button = Button(dialog, text="Cancel", command=on_cancel)
    cancel_button.pack(side="right", padx=20)
    
    dialog.mainloop()


def show_selection_window():
    """
    Display a full-screen window with a screenshot of the current screen,
    allowing the user to select a rectangular area by clicking and dragging.
    Simulates transparency by displaying the screenshot on a canvas.
    Returns the selected region as (x, y, width, height) or None if cancelled.
    """
    root = tk.Tk()
    root.overrideredirect(True)  # Remove window borders

    # Get screen dimensions
    logical_width = root.winfo_screenwidth()
    logical_height = root.winfo_screenheight()
    root.geometry(f"{logical_width}x{logical_height}+0+0")  # Full screen

    # Capture the screenshot
    screenshot = pyautogui.screenshot().convert('RGB')
    screenshot = screenshot.resize((logical_width, logical_height), Image.Resampling.LANCZOS)

    # Create a canvas for displaying the screenshot
    canvas = tk.Canvas(root, width=logical_width, height=logical_height, highlightthickness=0)
    canvas.pack()

    # Display the screenshot on the canvas
    photo = ImageTk.PhotoImage(screenshot)
    canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.image = photo  # Prevent garbage collection

    # Variables for selection
    start_x, start_y, rect = None, None, None
    selected_region = None

    def on_mouse_down(event):
        """Start drawing the selection rectangle."""
        nonlocal start_x, start_y, rect
        start_x, start_y = event.x, event.y
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)

    def on_mouse_move(event):
        """Update selection rectangle while dragging."""
        if rect:
            canvas.coords(rect, start_x, start_y, event.x, event.y)

    def on_mouse_up(event):
        """Finalize selection on mouse release."""
        nonlocal selected_region
        end_x, end_y = event.x, event.y
        x1, y1 = min(start_x, end_x), min(start_y, end_y)
        width, height = abs(end_x - start_x), abs(end_y - start_y)

        # Ensure the selection is larger than a minimal size
        if width > 10 and height > 10:
            selected_region = (x1, y1, width, height)
            print("Selected region:", selected_region)
            root.destroy()  # Close window after selection
            root.update()
        else:
            print("Selection too small, try again.")
            canvas.delete(rect)

    def on_double_click(event):
        """Exit the selection mode when the user double-clicks anywhere."""
        nonlocal selected_region
        selected_region = None  # Cancel selection
        print("Selection cancelled.")
        root.destroy()

    # Bind events
    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    canvas.bind("<Double-Button-1>", on_double_click)  # 🔥 Double-click to exit

    # Start the event loop
    root.mainloop()

    return selected_region

def activate_window(region):
    # region format: (x, y, width, height)
    x, y, width, height = region
    pyautogui.click(x, y)


def capture_screen(region):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

def detect_pacman(screen):
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pacman_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(pacman_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None

def play_pacman():
    global running
    print("Starting Pac-Man in region:", game_region)

    import random
    while running:
        screen = capture_screen(game_region)
        pacman_pos = detect_pacman(screen)
        if pacman_pos:
            list1 = ["up","down","left","right"]
            key1 = random.choice(list1)
            pyautogui.press(key1)
            print(key1)
        else:
            print("Pac-Man not detected!")
            break

        time.sleep(0.3)

# Main execution
show_initial_dialog()

if running:
    time.sleep(0.1)
    game_region = show_selection_window()

if game_region and running:
    activate_window(game_region)
    play_pacman()
else:
    print("No valid selection made or program cancelled.")
