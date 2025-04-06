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
import gc

from DQN_color import DQN

class PacmanServer:
    def __init__(self, host="0.0.0.0", port=9999, model_path="pacman_dqn.pth"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            #"mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        print(f"Using device: {self.device}")
        torch.set_default_dtype(torch.float32)
        
        self.template_ready = [cv2.imread("sample_ready.jpg", cv2.IMREAD_GRAYSCALE)]
        self.template_gameover = [cv2.imread("sample_gameover.jpg", cv2.IMREAD_GRAYSCALE),
                                 cv2.imread("sample_gameover2.jpg", cv2.IMREAD_GRAYSCALE)]
        
        self.actions = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.actions)
        self.frame_height = 168
        self.frame_width = 168
        self.model_path = model_path
        
        self.policy_net = DQN((12, self.frame_height, self.frame_width), self.num_actions).to(self.device)
        self.target_net = DQN((12, self.frame_height, self.frame_width), self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)  # Lowered LR
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
        self.step_count = 0
        self.previous_score = 0
        self.frames = deque(maxlen=8)
        self.log_file = "training_analysis.log"
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print(f"Server listening on {host}:{port}")
        if self.device.type in ["cuda", "mps"]:
            dummy = torch.rand(1, 12, 168, 168).to(self.device)
            self.policy_net(dummy)
            print(f"{self.device.type.upper()} warmed up")

    def preprocess_frame(self, screen):
        rgb = screen / 255.0
        return torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1)

    def get_state(self):
        if len(self.frames) < 4:
            padding = [torch.zeros(3, self.frame_height, self.frame_width, dtype=torch.float32, device=self.device)] * (4 - len(self.frames))
            state_frames = padding + list(self.frames)
        else:
            state_frames = list(self.frames)[-4:]
        state = torch.cat(state_frames, dim=0)
        return state.to(self.device)

    def detect_ready(self, gray_screen):
        height, width = gray_screen.shape[:2]
        x = int(width * 0.5) #0.55+0.145
        y = int(height * 0.2) #0.25+0.024
        roi_width = int(width * 0.2)
        roi_height = int(height * 0.08)
        
        roi = gray_screen[y:y + roi_height, x:x + roi_width]
        match_threshold = 0.8
        
        for template in self.template_ready:
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            match_score = np.max(result)
            print(f"READY Match Score: {match_score:.3f}")
            
            if match_score > match_threshold:
                return True
        return False

    def detect_game_over(self, gray_screen):
        height, width = gray_screen.shape[:2]
        x = int(width * 0.5)
        y = int(height * 0.2)
        roi_width = int(width * 0.2)
        roi_height = int(height * 0.08)
        
        roi = gray_screen[y:y + roi_height, x:x + roi_width]
        match_threshold = 0.8
        
        for template in self.template_gameover:
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
            q_values = self.policy_net(state.unsqueeze(0))
            return self.actions[q_values.max(1)[1].item()]

    def store_experience(self, state, action, reward, next_state, done):
        action_idx = self.actions.index(action)
        self.memory.append((state, action_idx, float(reward), next_state, done))  # Store as float (Python float32)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        gc.disable()
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        loss.backward()
        self.optimizer.step()
        gc.enable()

    def save_model(self, step):
        base, ext = os.path.splitext(self.model_path)
        save_path = f"{base}_{step}{ext}"
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def analyze_q_values(self):
        if len(self.memory) < self.batch_size:
            print(f"Step {self.step_count}: Not enough memory for analysis ({len(self.memory)}/{self.batch_size})")
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            policy_q_values = self.policy_net(states)
            target_q_values = rewards + (1 - dones) * self.gamma * self.target_net(next_states).max(1)[0]
            target_q_values = target_q_values.unsqueeze(1)
        
        q_mean = policy_q_values.mean().item()
        q_std = policy_q_values.std().item()
        q_max = policy_q_values.max().item()
        q_diff_mean = torch.abs(policy_q_values - target_q_values).mean().item()
        action_consistency = (policy_q_values.argmax(dim=1) == self.target_net(states).argmax(dim=1)).float().mean().item()
        hypothetical_loss = nn.MSELoss()(policy_q_values, target_q_values).item()
        
        log_message = (
            f"Step {self.step_count}:\n"
            f"Q-Value Mean: {q_mean:.4f}\n"
            f"Q-Value Std: {q_std:.4f}\n"
            f"Q-Value Max: {q_max:.4f}\n"
            f"Policy-Target Q-Diff Mean: {q_diff_mean:.4f}\n"
            f"Action Consistency: {action_consistency:.3f}\n"
            f"Hypothetical Loss: {hypothetical_loss:.4f}\n"
        )
        if q_std < 0.5 and q_diff_mean < 1.0 and hypothetical_loss < 0.1:
            log_message += "Q-values appear to be converging (stable and aligned).\n"
        else:
            log_message += "Q-values may not be converging (high variance, difference, or loss).\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.policy_net.load_state_dict(torch.load(self.model_path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = max(self.epsilon_min, self.epsilon_decay ** self.step_count)
            print(f"Model loaded from {self.model_path}, epsilon set to {self.epsilon:.3f}")
        else:
            print(f"No model found at {self.model_path}, starting fresh")

    def save_frames_to_files(self, prefix="frame"):
        if not self.frames:
            print("No frames to save.")
            return
        
        save_dir = "debug_frames"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for i, frame in enumerate(self.frames):
            frame_np = frame.permute(1, 2, 0).cpu().numpy() * 255
            frame_np = frame_np.astype(np.uint8)
            filename = f"{save_dir}/{prefix}_{timestamp}_{i:03d}.jpg"
            cv2.imwrite(filename, frame_np)

    def punish_death(self, num_pos):
        mem_len = len(self.memory)
        if mem_len == 0:
            return  # Nothing to punish if empty
        
        cut_pos = num_pos if mem_len >= num_pos else mem_len
        punish_idx = mem_len - cut_pos
        # Debug: Verify memory state
        print(f"mem_len={mem_len}, cut_pos={cut_pos}, punish_idx={punish_idx}")
        # Punish the target frame
        state, action_idx, _, next_state, _ = self.memory[punish_idx]
        self.memory[punish_idx] = (state, action_idx, -100, next_state, True)
        # Pop post-death frames from the end
        num_to_pop = mem_len - (punish_idx + 1)  # e.g., 35 - (31 + 1) = 3
        for _ in range(num_to_pop):
            self.memory.pop()  # Remove last entry
        #self.save_frames_to_files(f"death_{num_pos}_{time.time()}")

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
            self.previous_score = 0
            score_count = 0
            self.frames.clear()
            #gray_last = None
            waiting_for_reset = False
            
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
                        print(f"Client {addr} disconnected (data mismatch)")
                        break
                    
                    score_str, img_data = data.split(b"|", 1)
                    score = int(score_str.decode('utf-8'))
                    screen = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    if screen is None:
                        break

                    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                    if self.detect_ready(gray_screen):
                        print(f"Step {self.step_count}: (Ready) detected")
                        if not waiting_for_reset:
                            self.punish_death(8)
                        waiting_for_reset = False
                        self.frames.clear()
                        previous_state = None
                        previous_action = None
                        score_count = 0
                        conn.sendall("wait".encode('utf-8'))
                        continue
                    elif self.detect_game_over(gray_screen):
                        print(f"Step {self.step_count}: (Game Over) detected")
                        if not waiting_for_reset:
                            self.punish_death(8)
                        waiting_for_reset = False
                        self.frames.clear()
                        previous_state = None
                        previous_action = None
                        score_count = 0
                        self.previous_score = 0
                        conn.sendall("restart".encode('utf-8'))
                        continue

                    #if gray_last is not None and not waiting_for_reset:
                    #    diff = np.abs(gray_screen - gray_last)
                    #    changed_pixels = np.sum(diff > 10)
                    #    if changed_pixels < 100:
                    #        print(f"Step {self.step_count}: Freeze detected ({changed_pixels})")
                    #        self.punish_death(6)
                    #        waiting_for_reset = True
                    #        conn.sendall("skip".encode('utf-8'))
                    #        #cv2.imwrite(f"detect_pacman_{self.step_count}.jpg", gray_screen) #debug
                    #        continue

                    #gray_last = gray_screen
                    
                    #if waiting_for_reset:
                    #    conn.sendall("skip".encode('utf-8'))
                    #    continue
                    
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
                            reward = score_delta * 2
                            score_count = 0
                        else:
                            score_count += 1
                            reward = -1 if score_count > 10 else 0.1
                        
                        self.previous_score = score
                        print(f"Step {self.step_count}, Score: {score}, Reward: {reward}")

                        self.store_experience(previous_state, previous_action, reward, s_t, False)

                        if mode == "train":
                            t_start = time.time()
                            self.train()
                            t_end = time.time()
                            self.step_count += 1
                            print(f"Step {self.step_count-1}, Epsilon: {self.epsilon:.3f}, Train time: {t_end-t_start:.3f}s")
                            # Gradient target update
                            target_update_freq = 5000 if self.step_count <= 10000 else 1000
                            if self.step_count % target_update_freq == 0:
                                self.target_net.load_state_dict(self.policy_net.state_dict())
                            elif self.step_count % 1001 == 0:
                                self.save_model(self.step_count)
                                self.analyze_q_values()
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
    parser.add_argument("-m", "--model-path", type=str, default="pacman_dqn_c.pth",
                        help="Path to the model file (e.g., pacman_dqn.pth)")
    parser.add_argument("--mode", type=str, choices=["train", "play"], default="train",
                        help="Run mode: 'train' or 'play' (default: train)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    server = PacmanServer(host="0.0.0.0", port=9999, model_path=args.model_path)
    server.run(mode=args.mode)