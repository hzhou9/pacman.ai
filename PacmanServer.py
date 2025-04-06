import socket
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import os
import argparse
import random

from DQN import DQN

class PacmanServer:
    def __init__(self, host="0.0.0.0", port=9999, model_path="pacman_dqn.pth"):
        # Load the template (adjust path as needed)
        self.template_ready = [cv2.imread("sample_ready.jpg", cv2.IMREAD_GRAYSCALE)]
        self.template_gameover = [cv2.imread("sample_gameover.jpg", cv2.IMREAD_GRAYSCALE),cv2.imread("sample_gameover2.jpg", cv2.IMREAD_GRAYSCALE)]
        
        self.actions = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.actions)
        self.frame_height = 168
        self.frame_width = 168
        self.model_path = model_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN((4, self.frame_height, self.frame_width), self.num_actions).to(self.device)
        self.target_net = DQN((4, self.frame_height, self.frame_width), self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 5000
        self.step_count = 0
        self.previous_score = 0
        self.frames = deque(maxlen=8)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print(f"Server listening on {host}:{port}")

    def preprocess_frame(self, screen):
        gray = screen / 255.0
        return torch.FloatTensor(gray).unsqueeze(0)  # Shape: [1, 168, 168]

    def get_state(self):
        # Always use the most recent 4 frames, even if more are stored
        if len(self.frames) < 4:
            padding = [torch.zeros_like(self.frames[0])] * (4 - len(self.frames))
            state_frames = padding + list(self.frames)
        else:
            state_frames = list(self.frames)[-4:]  # Take the last 4 frames
        state = torch.cat(state_frames, dim=0)  # Shape: [4, 168, 168]
        return state.unsqueeze(0).to(self.device)  # Shape: [1, 4, 168, 168]
    
    def detect_ready(self, screen):
        # Define ROI based on your coordinates
        height, width = screen.shape[:2]
        x = int(width / 1.8)
        y = int(height / 4)
        roi_width = int(width / 6.9)
        roi_height = int(height / 41)
        
        # Extract ROI from grayscale screen
        roi = screen[y:y + roi_height, x:x + roi_width]

        #roi = cv2.equalizeHist(roi)
        # Threshold for match (0.7–0.9 typically works; adjust based on test)
        match_threshold = 0.65
        # Resize template to match ROI if necessary
        for template in self.template_ready:
            if template.shape != (roi_height, roi_width):
                template = cv2.resize(template, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
        
            # Perform template matching
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            match_score = np.max(result)
            print(f"READY Match Score: {match_score:.3f}")

            #debug
            #if match_score > 0:
            #    cv2.imwrite(f"ready.{time.time()}.jpg", roi)

            if match_score > match_threshold:
                return True

        return False

    def detect_game_over(self, screen):
        # Define ROI based on your coordinates
        height, width = screen.shape[:2]
        x = int(width / 1.8)
        y = int(height / 4)
        roi_width = int(width / 6.9)
        roi_height = int(height / 41)
        
        # Extract ROI from grayscale screen
        roi = screen[y:y + roi_height, x:x + roi_width]
        #roi = cv2.equalizeHist(roi)
        # Threshold for match (0.7–0.9 typically works; adjust based on test)
        match_threshold = 0.65
        # Resize template to match ROI if necessary
        for template in self.template_gameover:
            if template.shape != (roi_height, roi_width):
                template = cv2.resize(template, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
        
            # Perform template matching
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            match_score = np.max(result)
            print(f"GAMEOVER Match Score: {match_score:.3f}")

            if match_score > match_threshold:
                return True

        return False

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return self.actions[q_values.max(1)[1].item()]

    def store_experience(self, state, action, reward, next_state, done):
        action_idx = self.actions.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, step):
        base, ext = os.path.splitext(self.model_path)
        save_path = f"{base}_{step}{ext}"
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.policy_net.load_state_dict(torch.load(self.model_path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}, starting fresh")

    def save_frames_to_files(self, prefix="frame"):
        """Save all frames in self.frames to local image files."""
        if not self.frames:
            print("No frames to save.")
            return
        
        # Create a directory for saving frames if it doesn't exist
        save_dir = "debug_frames"
        os.makedirs(save_dir, exist_ok=True)
        
        # Get current timestamp for uniqueness
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save each frame in self.frames
        for i, frame in enumerate(self.frames):
            # Convert frame from tensor to numpy array
            frame_np = frame.squeeze(0).cpu().numpy() * 255  # Remove channel dim and scale back to 0-255
            frame_np = frame_np.astype(np.uint8)  # Convert to uint8 for OpenCV
            
            # Generate filename
            filename = f"{save_dir}/{prefix}_{timestamp}_{i:03d}.jpg"
            
            # Save the frame as an image
            cv2.imwrite(filename, frame_np)

    def run(self, mode="train"):
        if mode == "play":
            self.load_model()
            self.epsilon = 0
        elif mode == "train":
            self.load_model()
        
        while True:
            conn, addr = self.sock.accept()
            print(f"Client connected from {addr}")
            
            previous_state = None
            previous_action = None
            self.previous_score = 0  # Track game score
            score_count = 0 #continuous 
            self.frames.clear()  # Reset
            last_screen = None
            waiting_for_reset = False  # New flag for discard phase
            last_transition = None  # Store last normal transition for backtracking
            
            try:
                while True:
                    size_data = conn.recv(4)
                    if not size_data or len(size_data) < 4:
                        print(f"Client {addr} disconnected (incomplete size data)")
                        break
                    size = int.from_bytes(size_data, byteorder='big')
                    
                    data = b""
                    while len(data) < size:
                        packet = conn.recv(size - len(data))
                        if not packet:
                            print(f"Client {addr} disconnected (incomplete data)")
                            break
                        data += packet
                    
                    if len(data) != size:
                        print(f"Client {addr} disconnected (data mismatch: expected {size}, got {len(data)})")
                        break
                    
                    # Parse score and screenshot
                    score_str, img_data = data.split(b"|", 1)
                    score = int(score_str.decode('utf-8'))
                    screen = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if screen is None:
                        break

                    # check Game Over and Ready
                    if self.detect_ready(screen):
                        print(f"Step {self.step_count}: (Ready) detected")
                        waiting_for_reset = False
                        self.frames.clear()  # Clear frames for new life
                        previous_state = None
                        previous_action = None
                        score_count = 0
                        # Keep self.previous_score
                        conn.sendall("wait".encode('utf-8'))
                        continue
                    elif self.detect_game_over(screen):
                        print(f"Step {self.step_count}: (Game Over) detected")
                        waiting_for_reset = False
                        self.frames.clear()
                        previous_state = None
                        previous_action = None
                        score_count = 0
                        self.previous_score = 0  # Reset score for new game
                        conn.sendall("restart".encode('utf-8'))
                        continue

                    if last_screen is not None and not waiting_for_reset:
                        # check screen freeze, which indicate that pacman dies
                        diff = np.abs(screen - last_screen)
                        changed_pixels = np.sum(diff > 10)
                        if changed_pixels < 100:
                            print(f"Step {self.step_count}: Freeze detected ({changed_pixels}), assuming death")
                            if last_transition is not None:
                                # replace the params of last action which caused death
                                state, action, _, next_state = last_transition
                                action_idx = self.actions.index(action)
                                if self.memory:
                                    self.memory[-1] = (state, action_idx, -50, next_state, True)
                                else:
                                    self.store_experience(state, action, -50, next_state, True)
                            waiting_for_reset = True
                            conn.sendall("skip".encode('utf-8'))
                            last_screen = screen
                            continue

                    last_screen = screen
                    
                    # skip the frames from death to Game Over or Ready
                    if waiting_for_reset:
                        conn.sendall("skip".encode('utf-8'))
                        continue
                    # start to work on normal game frames
                    frame = self.preprocess_frame(screen)
                    self.frames.append(frame)
                    try:
                        s_t = self.get_state()
                    except Exception as e:
                        print(f"State error with client {addr}: {e}")
                        break
                    
                    if previous_state is not None:
                        score_delta = score - self.previous_score
                        if score_delta > 0:
                            reward = score_delta
                            score_count = 0
                        else:
                            score_count += 1
                            reward = -1 if score_count > 10 else 0.1
                                
                        self.previous_score = score
                        print(f"Step {self.step_count}, Score: {score}, Reward: {reward}")

                        last_transition = (previous_state, previous_action, reward, s_t)
                        self.store_experience(previous_state, previous_action, reward, s_t, False)

                        if mode == "train":
                            self.train()
                            self.step_count += 1
                            print(f"Step {self.step_count-1}, Epsilon: {self.epsilon:.3f}")
                            if self.step_count % self.target_update == 0:
                                self.target_net.load_state_dict(self.policy_net.state_dict())
                            if self.step_count % 1000 == 0:
                                self.save_model(self.step_count)
                            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                        
                        a_t = self.choose_action(s_t)
                        conn.sendall(a_t.encode('utf-8'))
                        previous_state = s_t
                        previous_action = a_t
                            
                    else:
                        self.previous_score = score
                        a_t = self.choose_action(s_t)
                        conn.sendall(a_t.encode('utf-8'))
                        previous_state = s_t
                        previous_action = a_t

                    time.sleep(0.01)
            
            except Exception as e:
                print(f"Error with client {addr}: {e}")
            finally:
                conn.close()
                print(f"Client {addr} disconnected")

def parse_args():
    parser = argparse.ArgumentParser(description="Pac-Man Server")
    parser.add_argument("-m", "--model-path", type=str, default="pacman_dqn.pth",
                        help="Path to the model file (e.g., pacman_dqn.pth)")
    parser.add_argument("--mode", type=str, choices=["train", "play"], default="train",
                        help="Run mode: 'train' or 'play' (default: train)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    server = PacmanServer(host="0.0.0.0", port=9999, model_path=args.model_path)
    server.run(mode=args.mode)