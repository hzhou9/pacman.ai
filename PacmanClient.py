import socket
import cv2
import pyautogui
import numpy as np
import time

import tkinter as tk
from tkinter import Canvas, Label, Button
from PIL import Image, ImageTk
import easyocr

import mss

class PacmanClient:
    def __init__(self, server_host="localhost", server_port=9999):
        self.game_region = None
        self.restart_target = None
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])
        self.sct = mss.mss()  # Initialize mss
        self.last_score = 0
        self.server_host = server_host
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_host, self.server_port))
        print(f"Connected to server at {server_host}:{server_port}")
    
    def show_initial_dialog(self):
        isNext = False
        try:
            dialog = tk.Tk()
        except Exception as e:
            print(f"Failed to initialize Tkinter: {e}")
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
            nonlocal isNext
            isNext = True
            dialog.destroy()
            dialog.update()
        
        def on_cancel():
            dialog.destroy()
        
        confirm_button = Button(dialog, text="Confirm", command=on_confirm)
        confirm_button.pack(side="left", padx=20)
        cancel_button = Button(dialog, text="Cancel", command=on_cancel)
        cancel_button.pack(side="right", padx=20)
        
        dialog.mainloop()
        return isNext
    

    def show_selection_window(self):
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

        # Variables for selection and text display
        start_x, start_y, rect, coord_text = None, None, None, None

        def on_mouse_down(event):
            if self.game_region:
                self.restart_target = (event.x, event.y)
                print("Restart target:", self.restart_target)
                root.destroy()
                root.update()
            else:
                """Start drawing the selection rectangle."""
                nonlocal start_x, start_y, rect, coord_text
                start_x, start_y = event.x, event.y
                rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)
                coord_text = canvas.create_text(start_x + 10, start_y - 10, text="X: 0, Y: 0", 
                                          fill='white', font=('Arial', 12), anchor='nw')

        def on_mouse_move(event):
            """Update selection rectangle while dragging."""
            nonlocal start_x, start_y, rect, coord_text
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)
                # Calculate and update relative coordinates
                rel_x = event.x - start_x
                rel_y = event.y - start_y
                # Update the text on the canvas
                canvas.itemconfig(coord_text, text=f"X: {rel_x}, Y: {rel_y}")

        def on_mouse_up(event):
            """Finalize selection on mouse release."""
            nonlocal start_x, start_y, rect, coord_text
            end_x, end_y = event.x, event.y
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            width, height = abs(end_x - start_x), abs(end_y - start_y)

            # Ensure the selection is larger than a minimal size
            if width > 10 and height > 10:
                self.game_region = (x1, y1, width, height)
                print("Selected region:", self.game_region)
                canvas.itemconfig(coord_text, text=f"[Last Step] Click the position of Game Restart Button")
            else:
                print("Selection too small, try again.")
                canvas.delete(rect)
                rect = None
                if coord_text:
                    canvas.delete(coord_text)
                    coord_text = None


        def on_double_click(event):
            """Exit the selection mode when the user double-clicks anywhere."""
            root.destroy()
            root.update()
            
        # Bind events
        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)
        canvas.bind("<Double-Button-1>", on_double_click)  # 🔥 Double-click to exit
        

        # Start the event loop
        root.mainloop()


    def activate_game_window(self):
        # region format: (x, y, width, height)
        x, y, width, height = self.game_region
        pyautogui.click(x + int(width/2), y + int(height/2))

    def capture_screen(self):
        #screenshot = pyautogui.screenshot(region=self.game_region)
        x, y, width, height = self.game_region
        region = {"top": y, "left": x, "width": width, "height": height}
        screenshot = self.sct.grab(region)
        return np.array(screenshot)

    def extract_score(self, screen):
        """Extract score from the top-left region of the screenshot with enhanced OCR for pixelated fonts."""
        # Crop score area relative to game_region
        height, width = screen.shape[:2]
        top = int(height / 24)
        left = int(width / 51)
        bottom = int(height / 14) + 2
        right = int(width / 6)
        score_region = screen[top:bottom, left:right]

        if np.all(score_region < 10):  # Threshold of 10 to account for minor noise
            print("Score area all black, skipping OCR")
            return -1

        # Enhance contrast
        gray = cv2.equalizeHist(score_region)

        # Resize for better resolution (triple size for clarity)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Invert colors for better OCR readability
        # gray = cv2.bitwise_not(gray)
        
        # Perform OCR
        result = self.reader.readtext(gray, detail=0)

        # Try to extract digits, handle pixelated font issues
        try:
            # Filter for digits and handle common pixelation errors
            cleaned_text = ''.join(filter(str.isdigit, ''.join(result)))
            score = int(cleaned_text) if cleaned_text else -1
            #print(f"SCORE: {score} (Raw Text: '{result}', Cleaned: '{cleaned_text}')")
        except ValueError:
            print(f"OCR ValueError: {result}")
            score = -1  # Default if OCR fails

        if score < 0:
            t = time.time()
            # Debug: Save processed image
            cv2.imwrite(f"score_region_debug.{t}.{result}.jpg", gray)
        else:
            #fix wrong OCR
            if score - self.last_score >= 100000:
                score = int(self.last_score/100000)*100000 + score%100000
            if score - self.last_score >= 10000:
                score = int(self.last_score/10000)*10000 + score%10000
            if score - self.last_score >= 1000:
                score = int(self.last_score/1000)*1000 + score%1000
            if score - self.last_score >= 100:
                score = int(self.last_score/100)*100 + score%100
            if score < self.last_score:
                score = self.last_score
            else:
                self.last_score = score

        return score

    def send_screenshot(self, screen):
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        # Step 2: Create a square image (gray_square) using the larger dimension
        height, width = gray.shape[:2]
        side_length = max(height, width)  # Use larger of height or width
        gray_square = np.zeros((side_length, side_length), dtype=np.uint8)  # Black background
        gray_square[0:height, 0:width] = gray  # Place original at top-left
        
        #print(f"Gray Square Shape: {gray_square.shape}, Min: {gray_square.min()}, Max: {gray_square.max()}")  # Debug
        
        # Step 3: Use gray_square for score detection
        score = self.extract_score(gray_square)
        
        # Step 4: Scale gray_square to 168x168 for sending to server
        if score >= 0:
            resized = cv2.resize(gray_square, (168, 168), interpolation=cv2.INTER_AREA)
            #print(f"Sent Image Shape: {resized.shape}, Min: {resized.min()}, Max: {resized.max()}")  # Debug
            
            # Encode as JPEG (grayscale)
            _, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            img_data = encoded.tobytes()
            
            # Combine score and image data
            score_str = f"{score:06d}"
            data = score_str.encode('utf-8') + b"|" + img_data
            size = len(data)
            self.sock.sendall(size.to_bytes(4, byteorder='big'))
            self.sock.sendall(data)
            return True
        return False

    def receive_action(self):
        # Receive action (assume 10-byte max string, e.g., "left")
        action = self.sock.recv(10).decode('utf-8').strip()
        return action

    def execute_action(self, action):
        if action == "restart":
            # Simulate Command + R on Mac to refresh the page
            print("restart")
            x, y = self.restart_target
            pyautogui.click(x, y)
            self.last_score = 0
            time.sleep(3)
            self.activate_game_window()
        elif action == "wait":
            time.sleep(3)
        elif action == "skip":
            time.sleep(0.1)
        else:
            # Execute normal movement actions
            pyautogui.press(action)

    def run(self):
        try:
            if self.show_initial_dialog():
                time.sleep(0.1)
                self.show_selection_window()
                if self.game_region:
                    self.activate_game_window()
                    while True:
                        t1 = time.time()
                        screen = self.capture_screen()
                        if not self.send_screenshot(screen):
                            time.sleep(0.1)
                            continue
                        t2 = time.time()
                        action = self.receive_action()
                        if not action:  # Check for empty action
                            print("Received empty action from server, retrying")
                            continue
                        t3 = time.time()
                        self.execute_action(action)
                        print(f"Screenshot: {t2-t1} | {action}: {t3-t2}")
                        #time.sleep(0.1)
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            self.sock.close()

if __name__ == "__main__":
    client = PacmanClient(server_host="127.0.0.1", server_port=9999)
    client.run()