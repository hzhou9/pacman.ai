import cv2
import numpy as np

def detect_game_over(screen):
        """Detect 'GAME OVER' using template matching at specific position."""
        # Load the template (adjust path as needed)
        template = cv2.imread("sample_gameover.jpg", cv2.IMREAD_GRAYSCALE)
        if template is None:
            print("Error: Could not load game_over.jpg")
            return False
        
        # Convert screen to grayscale
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # Define ROI based on your coordinates
        height, width = screen.shape[:2]
        x = int(width / 1.8)
        y = int(height / 4)
        roi_width = int(width / 6.9)
        roi_height = int(height / 41)
        
        # Extract ROI from grayscale screen
        roi = gray_screen[y:y + roi_height, x:x + roi_width]
        
        # Resize template to match ROI if necessary
        if template.shape != (roi_height, roi_width):
            template = cv2.resize(template, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
        
        # Perform template matching
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        
        # Threshold for match (0.7–0.9 typically works; adjust based on test)
        match_threshold = 0.8
        match_score = np.max(result)
        print(f"GAME OVER Match Score: {match_score:.3f}")
        
        return match_score > match_threshold

def test_detect_game_over(image_path="gameover.jpg"):
    """Load a screenshot and test the detect_game_over function."""
    # Load the screenshot
    screen = cv2.imread(image_path)
    if screen is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Ensure the image is resized to 168x168 (if not already)
    screen = cv2.resize(screen, (168, 168))
    
    # Test the detection
    is_game_over = detect_game_over(screen)

if __name__ == "__main__":
    # Specify your test image path here
    test_image_path = "gameover.jpg"
    test_detect_game_over(test_image_path)