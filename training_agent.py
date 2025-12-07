import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import socket
import time
import os
import signal
import threading
import csv
from datetime import datetime
import select
import warnings
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

from config_training import *

class CSVLogger:
    def __init__(self, log_dir=LOG_DIR, model_save_path=MODEL_SAVE_PATH):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        base_name = os.path.basename(model_save_path)
        self.fieldnames = [
            "global_step","episode","reward","throughput_kbps","packet_loss_rate",
            "rtt","rtt_ewma","rtt_slope","congestion"
        ]
        self.csv_file = os.path.join(log_dir, f"training_metrics_{base_name}.csv")

        if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
            print(f" Resuming CSV logging from: {self.csv_file}")
            try:
                with open(self.csv_file, 'r') as f:
                    rows = list(csv.reader(f))
                    last_line = rows[-1]
                    try:
                        self.global_step = int(last_line[0])
                        self.current_episode = int(last_line[1])
                        print(f" Resuming from step {self.global_step}, episode {self.current_episode}")
                    except (ValueError, IndexError):
                        print(" Could not parse last line of CSV, starting from zero.")
                        self.global_step = 0
                        self.current_episode = 0
                        with open(self.csv_file, "w", newline="") as f_write:
                            csv.DictWriter(f_write, fieldnames=self.fieldnames).writeheader()
            except Exception as e:
                print(f" Failed to read existing CSV: {e}. Recreating.")
                with open(self.csv_file, "w", newline="") as f_write:
                    csv.DictWriter(f_write, fieldnames=self.fieldnames).writeheader()
                self.global_step = 0
                self.current_episode = 0
        else:
            print(f" Creating new CSV log: {self.csv_file}")
            with open(self.csv_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
            self.global_step = 0
            self.current_episode = 0

    def log_step(self, data):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: data.get(k, "") for k in self.fieldnames})

    def increment_episode(self): self.current_episode += 1
    def increment_step(self): self.global_step += 1

def generate_scenario(seed=None, mode="mixed"):
    # deterministic option for testing:
    if seed is not None:
        rng = np.random.RandomState(seed)
        bandwidth_mbps = rng.uniform(5, 20)
        delay_ms = rng.uniform(20, 100)
        loss_rate = 0.0 if mode == "no_random_loss" else rng.uniform(0.0, 0.01)
        queue_size = int(max(int((bandwidth_mbps * 1e6) * (2 * delay_ms / 1000) / (1500 * 8)), 20))
    else:
        bandwidth_mbps = np.random.uniform(5, 20)
        delay_ms = np.random.uniform(20, 100)
        bdp_packets = (bandwidth_mbps * 1e6) * (2 * delay_ms / 1000) / (1500 * 8)
        min_queue = max(int(1.0 * bdp_packets), 20)
        max_queue = min(1000, max(min_queue * 5, 100))
        queue_size = np.random.randint(min_queue, max_queue)
        loss_rate = np.random.uniform(0.0, 0.01)  # 0% to 1% random loss

    print(f" Scenario: BW={bandwidth_mbps:.1f}Mbps, Delay={delay_ms:.0f}ms, Q={queue_size}p, Loss={loss_rate:.3f}")

    return {
        "bandwidth": f"{bandwidth_mbps:.1f}Mbps",
        "delay": f"{delay_ms:.0f}ms",
        "loss": loss_rate,
        "queue_size": f"{queue_size}p"
    }

class Ns3SimplerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ns3_script=NS3_SCRIPT, ns3_dir=NS3_DIR, csv_logger=None):
        super().__init__()
        self.ns3_script = ns3_script
        self.ns3_dir = ns3_dir
        self.csv_logger = csv_logger
        self.ns3_process = None
        self.sock = None
        self.recv_buffer = ""
        self.ns3_ready_event = threading.Event()

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(STATE_SIZE), high=np.ones(STATE_SIZE), dtype=np.float32)

        self.step_count = 0
        self.last_actions = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.last_rtt = 0.1  # initial value to compute variance

        self.last_loss_rate = 0.0

        # Sliding window for min RTT tracking
        self.rtt_history_window = []
        self.min_rtt_window_size = 20

        # reward/internal state
        self.rtt_history = []
        self.rtt_ewma = None
        self.congestion_state = False
        self.reward_step_count = 0
        self.measured_base_rtt = 0.1
        
        # Store scenario properties for relative normalization
        self.current_scenario_bw_mbps = 10.0  
        self.current_scenario_base_rtt_s = 0.1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()

        # generate scenario
        self.current_scenario = generate_scenario(seed=None)
        
        # Store scenario properties for relative normalization
        try:
            self.current_scenario_bw_mbps = float(self.current_scenario['bandwidth'].replace('Mbps', ''))
            delay_ms = float(self.current_scenario['delay'].replace('ms', ''))
            # Base RTT is 2x propagation delay
            self.current_scenario_base_rtt_s = (delay_ms * 2.0) / 1000.0
        except Exception as e:
            print(f"Warning: Could not parse scenario properties: {e}. Using defaults.")
            self.current_scenario_bw_mbps = 10.0
            self.current_scenario_base_rtt_s = 0.1
            self.measured_base_rtt = self.current_scenario_base_rtt_s

        if not self._start_ns3():
            raise RuntimeError("Failed to start ns-3")
        if not self._connect_to_ns3():
            raise ConnectionError("Failed to connect to ns-3")

        # reset counters and logger episode
        self.step_count = 0
        if self.csv_logger:
            self.csv_logger.increment_episode()

        # reset reward/detection state
        self.rtt_history_window = []
        self.rtt_history = []
        self.rtt_ewma = None
        self.congestion_state = False
        self.reward_step_count = 0
        self.last_actions = np.zeros(ACTION_SIZE, dtype=np.float32)

        self.last_loss_rate = 0.0

        state_str = self._await_state_and_send_action(np.zeros(self.action_space.shape))
        if state_str is None:
            raise ConnectionAbortedError("Failed warmup.")
            
        obs = self._parse_state_and_get_obs(state_str) 

        # initialize last_rtt and EWMA
        self.last_rtt = self._parse_raw_metrics(state_str)[2]
        self.rtt_history_window = [self.last_rtt] # Initialize with the first RTT
        self.rtt_ewma = self.last_rtt if self.rtt_ewma is None else self.rtt_ewma

        print(" Simplified Episode started successfully.")
        return obs, {}

    def step(self, action):
        self.step_count += 1
        state_str = self._await_state_and_send_action(action)
        if state_str is None:
            return np.zeros(self.observation_space.shape), 0.0, False, True, {}

        obs = self._parse_state_and_get_obs(state_str)
        tput, loss_count, rtt, _ = self._parse_raw_metrics(state_str)

        reward, loss_rate, diagnostics = self._calculate_reward(tput, loss_count, rtt)
        self.last_actions = action

        if self.csv_logger:
            self.csv_logger.increment_step()
            self.csv_logger.log_step({
                "global_step": self.csv_logger.global_step,
                "episode": self.csv_logger.current_episode,
                "reward": reward,
                "throughput_kbps": tput,
                "packet_loss_rate": loss_rate,
                "rtt": rtt,
                "rtt_ewma": diagnostics.get("rtt_ewma", ""),
                "rtt_slope": diagnostics.get("slope", ""),
                "congestion": int(diagnostics.get("congestion", 0))
            })

        EPISODE_LENGTH_STEPS = 1250 
        done = self.step_count >= EPISODE_LENGTH_STEPS
        
        if done:
            print(f" Episode {self.csv_logger.current_episode if self.csv_logger else '?'} finished.")
        return obs, reward, False, done, {}

    def _parse_raw_metrics(self, state_str):
        try:
            parts = state_str.split(',')
            return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            return 0.0, 0.0, 0.1, 0.0

    def _parse_state_and_get_obs(self, state_str):
        tput, loss, rtt, _ = self._parse_raw_metrics(state_str)

        norm_tput = np.clip(
            (tput / 1000.0) / (self.current_scenario_bw_mbps + 1e-6), 
            0, 
            1.5
        )
        norm_tput = (norm_tput - 0.75) / 0.75 
        

        
        queue_delay = max(0, rtt - self.measured_base_rtt)
        norm_queue_delay = np.tanh(queue_delay / 0.5) 

        norm_loss = float(np.clip(self.last_loss_rate, 0.0, 1.0))

        # RTT variance (jitter)
        rtt_variance = rtt - self.last_rtt
        self.last_rtt = rtt
        norm_rtt_variance = np.tanh(rtt_variance / 0.1)
        
        # return state size vector
        return np.array(
            [norm_tput,
            norm_loss,
            norm_queue_delay,
            norm_rtt_variance,
            float(self.last_actions[0]) if ACTION_SIZE > 0 else 0.0,
            float(self.last_actions[1]) if ACTION_SIZE > 1 else 0.0
        ], dtype=np.float32)

    def _calculate_reward(self, tput, loss_count, rtt):
        WARMUP_STEPS = 8
        MIN_TOTAL_PACKETS = 10
    
        self.reward_step_count = getattr(self, "reward_step_count", 0) + 1

        if self.reward_step_count > WARMUP_STEPS:
            self.rtt_history_window.append(rtt)
            if len(self.rtt_history_window) > self.min_rtt_window_size:
                self.rtt_history_window.pop(0)
        
        base_rtt = min(self.rtt_history_window) if len(self.rtt_history_window) > 0 else rtt
        self.measured_base_rtt = base_rtt

        PACKET_SIZE_BYTES = 1500.0
        interval_duration_seconds = RL_STEP_INTERVAL_MS / 1000.0 
        bytes_received = (tput * 1000.0 / 8.0) * interval_duration_seconds
        packets_received = bytes_received / PACKET_SIZE_BYTES
        total_packets = packets_received + loss_count

        if total_packets < MIN_TOTAL_PACKETS:
            loss_rate = 0.0
        else:
            loss_rate = loss_count / total_packets if total_packets > 0 else 0.0
        
        self.last_loss_rate = float(np.clip(loss_rate, 0.0, 1.0))

        throughput_mbps = tput / 1000.0

        queue_delay = max(0.0, rtt - self.measured_base_rtt)

        throughput_utility = np.log(throughput_mbps + 1.0)


        LATENCY_PENALTY_COEF = 5.0 
        latency_penalty = LATENCY_PENALTY_COEF * queue_delay
        
        LOSS_PENALTY_COEF = 10.0 
        loss_penalty = LOSS_PENALTY_COEF * self.last_loss_rate
        
        
        REWARD_SCALING_FACTOR = 10.0
        reward = REWARD_SCALING_FACTOR * (throughput_utility - latency_penalty - loss_penalty)

        reward = float(np.clip(reward, -20.0, 50.0)) 
        
        if self.step_count % 50 == 0:
             print(f"Step {self.step_count:3d} | R:{reward:6.2f} | "
               f"Tput_Util:{REWARD_SCALING_FACTOR * throughput_utility:6.2f} | "
               f"Lat_Pen:{-REWARD_SCALING_FACTOR * latency_penalty:6.2f} (QDelay:{queue_delay:.3f}s) | "
               f"Loss_Pen:{-REWARD_SCALING_FACTOR * loss_penalty:6.2f} (Rate:{self.last_loss_rate*100:5.1f}%) | "
               f"RTT:{rtt:.3f}s | BaseRTT:{self.measured_base_rtt:.3f}s")
        
        diagnostics = {
            "rtt_ewma": base_rtt,
            "slope": queue_delay,
            "base_rtt": base_rtt,
            "congestion_prob": 0.0,
            "congestion": int(queue_delay > (base_rtt * 0.4)),
            "utility": 0.0, # Utility is no longer used
        }
        
        return reward, self.last_loss_rate, diagnostics


    def _start_ns3(self):
        EPISODE_LENGTH_STEPS = 1250
        duration = int((EPISODE_LENGTH_STEPS * (RL_STEP_INTERVAL_MS / 1000.0)) + 50)
        
        cmd = [
            "./ns3", "run", f"scratch/{self.ns3_script}", "--",
            f"--bandwidth={self.current_scenario['bandwidth']}",
            f"--delay={self.current_scenario['delay']}",
            f"--error_p={self.current_scenario['loss']}",
            f"--queue_size={self.current_scenario['queue_size']}",
            f"--rl_step_interval={RL_STEP_INTERVAL_MS}", 
            f"--duration={duration}", 
            f"--rl_port={RL_PORT}", 
            "--enable_rl=true"
        ]
        try:
            self.ns3_ready_event.clear()
            self.ns3_process = subprocess.Popen(
                cmd, cwd=self.ns3_dir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, preexec_fn=os.setsid
            )
            threading.Thread(target=self._stream_reader, args=(self.ns3_process.stdout,), daemon=True).start()
            threading.Thread(target=self._stream_reader, args=(self.ns3_process.stderr,), daemon=True).start()
            return True
        except Exception as e:
            print(f" Failed to start ns-3: {e}")
            return False

    def _stream_reader(self, stream):
        try:
            for line in iter(stream.readline, ""):
                if "NS3READYFORCONNECTION" in line:
                    self.ns3_ready_event.set()
        except Exception:
            pass

    def _connect_to_ns3(self):
        if not self.ns3_ready_event.wait(timeout=CONNECTION_TIMEOUT):
            print(" Timeout waiting for ns-3.")
            return False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(10):
            try:
                self.sock.connect(("127.0.0.1", RL_PORT))
                print(" Socket connected!")
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(1.0)
        print(" Failed to connect after 10 attempts")
        return False

    def _await_state_and_send_action(self, action, timeout=10.0):
        try:
            if "\n" not in self.recv_buffer:
                ready, _, _ = select.select([self.sock], [], [], timeout)
                if not ready:
                    print(" Timeout waiting for state")
                    return None
                data = self.sock.recv(4096).decode(errors="ignore")
                if not data:
                    print(" Connection closed")
                    return None
                self.recv_buffer += data
            
            
            if "\n" not in self.recv_buffer:
                return None
                
            line, self.recv_buffer = self.recv_buffer.split("\n", 1)
            
            self.sock.sendall(f"{action[0]:.6f},{action[1]:.6f}\n".encode())
            return line.strip()
            
        except Exception as e:
            print(f"Error in await/send: {e}")
            return None

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.ns3_process and self.ns3_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.ns3_process.pid), signal.SIGKILL)
                self.ns3_process.wait(3)
            except Exception:
                pass
        self.ns3_process = None
        self.sock = None


def main():
    if not os.path.exists(NS3_DIR):
        print(f" ERROR: NS3_DIR='{NS3_DIR}' not found.")
        return

    latest_checkpoint = None
    if os.path.exists(LOG_DIR):
        checkpoints = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.startswith("rl_model_simple_") and f.endswith(".zip")]
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime)
            latest_checkpoint = checkpoints[-1]

    model_path = MODEL_SAVE_PATH + ".zip"
    if latest_checkpoint:
        model_path = latest_checkpoint

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    csv_logger = CSVLogger()
    env = Ns3SimplerEnv(csv_logger=csv_logger)

    callback = CheckpointCallback(5000, save_path=LOG_DIR, name_prefix="rl_model_simple")

    reset_timesteps = False
    if os.path.exists(model_path):
        print(f" Resuming training from saved model: {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log=LOG_DIR)
    else:
        print(" Creating new PPO model for simple training...")
        model = PPO(
            POLICY_TYPE, 
            env, 
            verbose=1, 
            tensorboard_log=LOG_DIR, 
            learning_rate=LEARNING_RATE, 
            n_steps=2048,         
            batch_size=64,        
            vf_coef=1.0,          
            max_grad_norm=0.5
        )
        reset_timesteps = True

    try:
        print(f"\n{'='*60}\n Starting/Resuming training (Total Steps: {TOTAL_TIMESTEPS})\n{'='*60}")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, reset_num_timesteps=reset_timesteps)
    except Exception as e:
        print(f"\n\n Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("\n Saving final simple model...")
        model.save(MODEL_SAVE_PATH)
        env.close()
        print(" Environment closed.")

if __name__ == "__main__":
    main()