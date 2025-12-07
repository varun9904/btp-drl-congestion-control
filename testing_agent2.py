import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import socket
import time
import os
import signal
import threading
import re
import pandas as pd
from tqdm import tqdm
import select
import warnings
import matplotlib.pyplot as plt

from scipy import stats
from math import sqrt

warnings.filterwarnings("ignore", message=".Gym has been unmaintained.")

from config_testing import * 
from stable_baselines3 import PPO

NUM_RUNS = 10

SCENARIOS_TO_PLOT = [
    "Fiber Optic (Ideal)",
    "Home WiFi (Stable, 1% Loss)",
    "Public WiFi (Unstable, 2% Loss)",
    "Cable Modem (Bufferbloat)",
    "Satellite Link (High Latency, 0.5% Loss)",
]

def get_real_params_from_actions(action_log):
    BETA_MIN, BETA_DEFAULT, BETA_MAX = 0.41, 0.7, 0.99
    C_MIN, C_DEFAULT, C_MAX = 0.01, 0.4, 0.79
    
    real_betas = []
    real_cs = []
    
    for action in action_log:
        beta_action, c_action = action[0], action[1]
        
        if beta_action >= 0:
            beta = BETA_DEFAULT + beta_action * (BETA_MAX - BETA_DEFAULT)
        else:
            beta = BETA_DEFAULT + beta_action * (BETA_DEFAULT - BETA_MIN)
        
        if c_action >= 0:
            c = C_DEFAULT + c_action * (C_MAX - C_DEFAULT)
        else:
            c = C_DEFAULT + c_action * (C_DEFAULT - C_MIN)
            
        real_betas.append(np.clip(beta, BETA_MIN, BETA_MAX))
        real_cs.append(np.clip(c, C_MIN, C_MAX))
        
    return real_betas, real_cs

PKT_SIZE_BYTES = 1500.0 
EPS = 1e-9

TEST_SCENARIOS = [
    {"name": "Fiber Optic (Ideal)", 
     "bandwidth": "100Mbps", 
     "delay": "10ms",
     "loss": 0.00, 
     "queue_size": "250p"},
    
    {"name": "Cable Modem (Bufferbloat)", 
     "bandwidth": "50Mbps", 
     "delay": "50ms",
     "loss": 0.00, 
     "queue_size": "2000p"},
    
    {"name": "Home WiFi (Stable, 1% Loss)", 
     "bandwidth": "40Mbps", 
     "delay": "50ms",
     "loss": 0.01, 
     "queue_size": "1000p"},
    
    {"name": "Public WiFi (Unstable, 2% Loss)", 
     "bandwidth": "40Mbps", 
     "delay": "50ms",
     "loss": 0.02, 
     "queue_size": "1000p"},
    
    {"name": "Satellite Link (High Latency, 0.5% Loss)", 
     "bandwidth": "20Mbps", 
     "delay": "250ms",
     "loss": 0.005, 
     "queue_size": "1600p"}
]

BASELINE_AGENTS = [
    {"name": "CUBIC", "tcp_prot": "ns3::TcpCubic", "color": "#D55E00", "linestyle": "-"}
]
AGENT_COLORS = {f"Baseline_{b['name']}": b['color'] for b in BASELINE_AGENTS}
AGENT_COLORS["RL"] = "#0072B2"

AGENT_LINESTYLES = {f"Baseline_{b['name']}": b['linestyle'] for b in BASELINE_AGENTS}
AGENT_LINESTYLES["RL"] = "-"

TEST_DURATION_S = 30

class TestNs3SimpleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ns3_script=NS3_SCRIPT, ns3_dir=NS3_DIR, fairness_mode=False):
        super().__init__()
        self.ns3_script = ns3_script
        self.ns3_dir = ns3_dir
        self.ns3_process = None
        self.sock = None
        self.recv_buffer = ""
        self.ns3_ready_event = threading.Event()
        self.debug_mode = True
        self.TEST_DURATION_S = TEST_DURATION_S
        self.fairness_mode = fairness_mode 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(STATE_SIZE), high=np.ones(STATE_SIZE), dtype=np.float32)

        self.ns3_stdout_lines = []
        self.ns3_stderr_lines = []
        self.action_log = []
        self.last_actions = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.last_rtt = 0.1
        self.last_loss_rate = 0.0 

        self.current_scenario_bw_mbps = 10.0
        self.current_scenario_base_rtt_s = 0.1
        
        self.measured_base_rtt = 0.1
        self.rtt_history_window = []
        self.min_rtt_window_size = 20
        self.reward_step_count = 0

    def reset(self, scenario, seed=None, options=None):
        super().reset(seed=seed)
        self.close()

        # FIX #4: Clear recv_buffer to prevent corruption
        self.recv_buffer = ""
        
        self.current_scenario = scenario
        self.current_scenario['seed'] = seed
        
        if self.fairness_mode:
            print(f"\n🧪 RL Fairness Test: '{scenario['name']}' | Seed: {seed}")
        else:
            print(f"\n🔧 RL Test: '{scenario['name']}' | Seed: {seed}")
        
        try:
            self.current_scenario_bw_mbps = float(scenario['bandwidth'].replace('Mbps', ''))
            delay_ms = float(scenario['delay'].replace('ms', ''))
            self.current_scenario_base_rtt_s = (delay_ms * 2.0) / 1000.0
        except Exception as e:
            print(f"Warning: Could not parse scenario properties: {e}. Using defaults.")
            self.current_scenario_bw_mbps = 10.0
            self.current_scenario_base_rtt_s = 0.1
        
        self.measured_base_rtt = self.current_scenario_base_rtt_s
        self.rtt_history_window = []
        self.reward_step_count = 0

        if not self._start_ns3(scenario):
            raise RuntimeError("Failed to start ns-3.")
        if not self._connect_to_ns3():
            self._print_ns3_output()
            raise ConnectionError("Failed to connect to ns-3.")

        self.action_log = []
        self.last_actions = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.last_loss_rate = 0.0 

        state_str = self._get_next_state(np.zeros(self.action_space.shape, dtype=np.float32))
        if state_str is None:
            self._print_ns3_output()
            raise ConnectionAbortedError("Failed warmup.")

        tput, loss, rtt, cwnd = self._parse_raw_metrics(state_str)
        self.last_rtt = rtt
        self.rtt_history_window = [rtt]
        self.measured_base_rtt = rtt
        
        obs = self._parse_state_and_get_obs(state_str, tput, loss, rtt)
        
        print(" RL episode starting!")
        return obs, {}

    def step(self, action):
        self.action_log.append(np.array(action, copy=True))
        state_str = self._get_next_state(action)

        if state_str is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

        tput, loss, rtt, cwnd = self._parse_raw_metrics(state_str)
        
        self.reward_step_count += 1
        WARMUP_STEPS = 8 
        if self.reward_step_count > WARMUP_STEPS:
            self.rtt_history_window.append(rtt)
            if len(self.rtt_history_window) > self.min_rtt_window_size:
                self.rtt_history_window.pop(0)
        
        if len(self.rtt_history_window) > 0:
            self.measured_base_rtt = min(self.rtt_history_window)
        else:
            self.measured_base_rtt = rtt
            
        obs = self._parse_state_and_get_obs(state_str, tput, loss, rtt)

        self.last_actions = np.array(action, copy=True)
        
        info = {'rtt': rtt, 'loss_count': loss, 'throughput': tput, 'cwnd': cwnd}
        return obs, 0.0, False, False, info

    def _start_ns3(self, scenario):
        duration = self.TEST_DURATION_S 
        command = [
            "./ns3", "run", f"scratch/{self.ns3_script}", "--",
            f"--bandwidth={scenario['bandwidth']}",
            f"--delay={scenario['delay']}",
            f"--error_p={scenario['loss']}",
            f"--queue_size={scenario['queue_size']}",
            f"--rl_step_interval={RL_STEP_INTERVAL_MS}", 
            f"--duration={duration}",
            f"--rl_port={RL_PORT}",
            "--enable_rl=true"
        ]

        if scenario.get('seed') is not None:
             command.append(f"--RngRun={scenario['seed']}")
        
        if self.fairness_mode:
            command.append("--competing_flow=true")
        
        print("🔧 Starting NS-3 with command:\n   ", " ".join(command))
        try:
            self.ns3_ready_event.clear()
            self.ns3_stdout_lines = []
            self.ns3_stderr_lines = []
            self.ns3_process = subprocess.Popen(
                command, cwd=self.ns3_dir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, preexec_fn=os.setsid
            )
            threading.Thread(target=self._stream_reader, args=(self.ns3_process.stdout, "stdout"), daemon=True).start()
            threading.Thread(target=self._stream_reader, args=(self.ns3_process.stderr, "stderr"), daemon=True).start()
            time.sleep(0.5)
            if self.ns3_process.poll() is not None:
                print(f" NS-3 process exited immediately with code {self.ns3_process.returncode}")
                self._print_ns3_output()
                return False
            print(" NS-3 process started successfully")
            return True
        except Exception as e:
            print(f" Failed to start ns-3: {e}")
            import traceback; traceback.print_exc()
            return False

    def _stream_reader(self, stream, stream_type):
        try:
            for raw_line in iter(stream.readline, ''):
                if not raw_line:
                    break
                line = raw_line.rstrip("\n")
                if stream_type == "stdout":
                    self.ns3_stdout_lines.append(line)
                else:
                    self.ns3_stderr_lines.append(line)
                
                normalized = re.sub(r'[\s_\-]+', '', line).upper()
                if "NS3READYFORCONNECTION" in normalized and not self.ns3_ready_event.is_set():
                    print(f" Detected official ready signal: {line}")
                    self.ns3_ready_event.set()
        except Exception:
            pass

    def _print_ns3_output(self):
        print("\n" + "="*60 + "\n📋 NS-3 STDOUT (last 20 lines):\n" + "="*60)
        for line in self.ns3_stdout_lines[-20:]:
            print(line)
        print("\n" + "="*60 + "\n📋 NS-3 STDERR (last 20 lines):\n" + "="*60)
        for line in self.ns3_stderr_lines[-20:]:
            print(line)
        print("="*60 + "\n")

    def _connect_to_ns3(self):
        print("⏳ Waiting for NS-3 official ready signal...")
        if not self.ns3_ready_event.wait(timeout=CONNECTION_TIMEOUT):
            print(" Timeout waiting for ns-3 signal")
            self._print_ns3_output()
            return False
        print(" Ready signal received!")
        if self.ns3_process and self.ns3_process.poll() is not None:
            print(" NS-3 died before connection")
            self._print_ns3_output()
            return False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        for attempt in range(15):
            try:
                self.sock.connect(("127.0.0.1", RL_PORT))
                self.sock.setblocking(False)
                print(f" Socket connected on attempt {attempt+1}!")
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(1.0)
        print(" Failed to connect after attempts.")
        self._print_ns3_output()
        return False

    def _parse_raw_metrics(self, state_str):
        try:
            parts = state_str.split(',')
            return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            return 0.0, 0.0, 0.1, 0.0

    def _parse_state_and_get_obs(self, state_str, tput, loss, rtt):
        norm_tput = np.clip(
            (tput / 1000.0) / (self.current_scenario_bw_mbps + 1e-6), 
            0, 1.5
        )
        norm_tput = (norm_tput - 0.75) / 0.75 
        
        queue_delay = max(0, rtt - self.measured_base_rtt)
        norm_queue_delay = np.tanh(queue_delay / 0.5) 

        interval_duration_seconds = RL_STEP_INTERVAL_MS / 1000.0 
        bytes_received = (tput * 1000.0 / 8.0) * interval_duration_seconds
        packets_received = bytes_received / PKT_SIZE_BYTES
        total_packets = packets_received + loss
        
        current_loss_rate = 0.0
        if total_packets >= 10:
            current_loss_rate = loss / total_packets if total_packets > 0 else 0.0
        
        self.last_loss_rate = float(np.clip(current_loss_rate, 0.0, 1.0))
        norm_loss = self.last_loss_rate

        rtt_variance = rtt - self.last_rtt
        self.last_rtt = rtt
        norm_rtt_variance = np.tanh(rtt_variance / 0.1)
        
        obs = np.array([
            norm_tput,
            norm_loss,
            norm_queue_delay,
            norm_rtt_variance,
            float(self.last_actions[0]) if ACTION_SIZE > 0 else 0.0,
            float(self.last_actions[1]) if ACTION_SIZE > 1 else 0.0
        ], dtype=np.float32)
        return obs

    def _get_next_state(self, action, timeout=8.0):
        try:
            action_str = f"{action[0]:.6f},{action[1]:.6f}\n"
            try:
                self.sock.sendall(action_str.encode())
            except (BrokenPipeError, OSError) as e:
                print(" Socket write failed:", e)
                return None

            deadline = time.time() + timeout
            while '\n' not in self.recv_buffer and time.time() < deadline:
                rready, _, _ = select.select([self.sock], [], [], 1.0)
                if not rready:
                    if self.ns3_process and self.ns3_process.poll() is not None:
                        print(" NS-3 process terminated unexpectedly while waiting for state.")
                        return None
                    continue
                try:
                    data = self.sock.recv(8192)
                except BlockingIOError:
                    continue
                if not data:
                    print(" Socket closed by ns-3")
                    return None
                self.recv_buffer += data.decode(errors='ignore')

            if '\n' not in self.recv_buffer:
                print(" Timeout waiting for ns-3 state")
                return None
            line, self.recv_buffer = self.recv_buffer.split('\n', 1)
            return line.strip()
        except Exception as e:
            print(f" Error in _get_next_state: {e}")
            return None

    def get_final_stats(self):
        if not self.ns3_process:
            return -1, -1 
        try:
            if self.sock:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                    self.sock.close()
                except Exception:
                    pass
            print(" Waiting for NS-3 to finish...")
            try:
                self.ns3_process.wait(timeout=60)
                print(" NS-3 completed normally")
            except subprocess.TimeoutExpired:
                print(" Timeout waiting for NS-3; killing...")
                try:
                    os.killpg(os.getpgid(self.ns3_process.pid), signal.SIGKILL)
                except Exception:
                    pass
                return -1, -1 
            
            all_output = "\n".join(self.ns3_stdout_lines + self.ns3_stderr_lines)
            
            rl_bytes = -1
            comp_bytes = -1

            rl_bytes_match = re.search(r"Total Rx Bytes \(RL Flow\) = (\d+)", all_output)
            if rl_bytes_match:
                rl_bytes = int(rl_bytes_match.group(1))
            else:
                bytes_match = re.search(r"Total Rx Bytes = (\d+)", all_output, re.IGNORECASE)
                if bytes_match:
                    rl_bytes = int(bytes_match.group(1))

            if self.fairness_mode:
                comp_bytes_match = re.search(r"Total Rx Bytes \(Competing Flow\) = (\d+)", all_output)
                if comp_bytes_match:
                    comp_bytes = int(comp_bytes_match.group(1))
                print(f" NS-3 finished: RL={rl_bytes:,} bytes, Competing={comp_bytes:,} bytes")
                return rl_bytes, comp_bytes
            else:
                print(f" NS-3 finished: {rl_bytes:,} bytes transferred")
                return rl_bytes, -1
            
        except Exception as e:
            print(f" Error getting final stats: {e}")
            return -1, -1

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


def run_baseline_test(scenario, ns3_dir, ns3_script, duration_s, baseline_tcp_prot):
    print(f"\n Running Baseline Test for '{scenario['name']}' (Agent: {baseline_tcp_prot})...")
    command = [
        "./ns3", "run", f"scratch/{ns3_script}", "--", 
        f"--bandwidth={scenario['bandwidth']}",
        f"--delay={scenario['delay']}",
        f"--error_p={scenario['loss']}",
        f"--queue_size={scenario['queue_size']}",
        f"--duration={duration_s}",
        f"--rl_step_interval={RL_STEP_INTERVAL_MS}",
        f"--baseline_tcp={baseline_tcp_prot}",  
        "--enable_rl=false"
    ]
    if scenario.get('seed') is not None:
         command.append(f"--RngRun={scenario['seed']}")
    print("   Command: " + " ".join(command))

    log_bs_throughputs = []
    log_bs_loss_counts = []
    log_bs_rtts = []
    log_bs_cwnds = []
    all_output_lines = [] 

    try:
        process = subprocess.Popen(
            command, cwd=ns3_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, preexec_fn=os.setsid
        )

        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if not line:
                break
            all_output_lines.append(line)

            if line.startswith("BASELINE_DATA,"):
                try:
                    parts = line.split(',')
                    log_bs_throughputs.append(float(parts[1]))
                    log_bs_loss_counts.append(float(parts[2]))
                    log_bs_rtts.append(float(parts[3]))
                    log_bs_cwnds.append(float(parts[4]))
                except Exception as e:
                    print(f"Baseline parse error: {e} on line: {line}")

        process.wait(timeout=duration_s * 3)

        all_output = "\n".join(all_output_lines)
        bytes_match = re.search(r"Total\s*Rx\s*Bytes\s*=\s*(\d+)", all_output, re.IGNORECASE)
        avg_rtt_match = re.search(r"Average\s*RTT\s*=\s*([\d\.]+)\s*s", all_output, re.IGNORECASE)
        total_loss_match = re.search(r"Total\s*Packet\s*Loss\s*=\s*(\d+)", all_output, re.IGNORECASE)

        if not bytes_match:
            print(" Baseline parsing failed!")
            print(all_output[-2000:])
            return {"total_bytes": -1, "avg_rtt_s": -1.0, "total_loss": -1}, pd.DataFrame()

        total_bytes = int(bytes_match.group(1))
        avg_rtt_s = float(avg_rtt_match.group(1)) if avg_rtt_match else 0.0
        total_loss = int(total_loss_match.group(1)) if total_loss_match else 0

        print(f" Baseline: {total_bytes:,} bytes, {avg_rtt_s*1000:.2f}ms RTT, {total_loss} loss")

        summary = {"total_bytes": total_bytes, "avg_rtt_s": avg_rtt_s, "total_loss": total_loss}

        df_bs = pd.DataFrame({
            'tput': log_bs_throughputs,
            'loss_count': log_bs_loss_counts,
            'rtt': log_bs_rtts,
            'cwnd': log_bs_cwnds
        })
        
        if len(df_bs) > 1:
            # Defensive: Check if loss looks cumulative
            total_loss_final = df_bs['loss_count'].iloc[-1]
            total_loss_start = df_bs['loss_count'].iloc[0]
            
            # If final loss is significantly higher than start, it's cumulative
            if total_loss_final > total_loss_start + 5:
                print("   (Converting cumulative loss to per-step loss)")
                df_bs['loss_count'] = df_bs['loss_count'].diff().fillna(0).clip(lower=0)

        return summary, df_bs

    except Exception as e:
        print(f" Baseline crashed: {e}")
        import traceback; traceback.print_exc()
        return {"total_bytes": -1, "avg_rtt_s": -1.0, "total_loss": -1}, pd.DataFrame()


def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    m, se = np.mean(a), stats.sem(a)
    
    if n <= 1 or se == 0:
      return m, 0.0
    
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def analyze_aggregate_results(all_results, all_fairness_results, scenario_durations):
    if not all_results:
        print("No results to analyze.")
        return

    df = pd.DataFrame(all_results)
    
    df['duration_s'] = df['scenario'].map(scenario_durations)
    df['mbps'] = np.maximum(0.0, (df['total_bytes'] * 8.0) / (df['duration_s'] * 1e6))
    df['avg_rtt_ms'] = df['avg_rtt_s'] * 1000.0
    df['recv_pkts'] = np.maximum(0.0, df['total_bytes'] / PKT_SIZE_BYTES)
    df['sent_pkts'] = df['recv_pkts'] + np.maximum(0.0, df['total_loss'])
    df['loss_rate'] = np.where(df['sent_pkts'] > 0, (df['total_loss'] / df['sent_pkts']) * 100.0, 0.0)

    print("\n" + "="*80 + "\n AGGREGATE TEST RESULTS (Mean ± 95% CI over " + str(NUM_RUNS) + " runs)\n" + "="*80)

    for name in df['scenario'].unique():
        print(f"\n Scenario: {name}\n" + "-"*70)
        s_data = df[df['scenario'] == name]
        
        rl_data = s_data[s_data['agent_type'] == 'RL']
        rl_tput = rl_data['mbps'].values
        
        agent_names = s_data['agent_type'].unique()
        
        print(f"    {'Agent':<12} | {'Throughput (Mbps)':<20} | {'RTT (ms)':<20} | {'Loss Rate (%)':<20}")
        print(f"    {'-'*12:<12} | {'-'*20:<20} | {'-'*20:<20} | {'-'*20:<20}")

        for agent in agent_names:
            agent_data = s_data[s_data['agent_type'] == agent]
            
            m_tput, ci_tput = mean_ci(agent_data['mbps'])
            m_rtt, ci_rtt = mean_ci(agent_data['avg_rtt_ms'])
            m_loss, ci_loss = mean_ci(agent_data['loss_rate'])
            
            print(f"    {agent:<12} | {m_tput:7.2f} ± {ci_tput:5.2f}       | {m_rtt:7.2f} ± {ci_rtt:5.2f}       | {m_loss:7.2f} ± {ci_loss:5.2f}")

        print("\n     Statistical Tests (vs RL Agent Throughput):")
        
        for agent in agent_names:
            if agent == 'RL':
                continue
            
            base_tput = s_data[s_data['agent_type'] == agent]['mbps'].values
            
            if len(rl_tput) > 1 and len(base_tput) > 1:
                try:
                    stat, p_value = stats.wilcoxon(rl_tput, base_tput, zero_method='zsplit')
                    
                    if p_value < 0.05:
                        diff_tput, _ = mean_ci(rl_tput - base_tput)
                        verdict = f"Significant (p={p_value:.4f}, diff={diff_tput:+.2f} Mbps)"
                    else:
                        verdict = f"Not Significant (p={p_value:.4f})"
                except ValueError as e:
                    verdict = f"Test Error (e.g., all values equal): {e}"
            else:
                verdict = "N/A (not enough data)"
                
            print(f"    vs. {agent:<10}: {verdict}")

    print("\n" + "="*80 + "\n AGGREGATE FAIRNESS RESULTS (Mean ± 95% CI over " + str(NUM_RUNS) + " runs)\n" + "="*80)
    
    if all_fairness_results:
        df_fair = pd.DataFrame(all_fairness_results)
        
        df_fair['total_bytes'] = df_fair['rl_agent_bytes'] + df_fair['competing_cubic_bytes']
        df_fair['rl_share_pct'] = np.where(
            df_fair['total_bytes'] > 0,
            (df_fair['rl_agent_bytes'] / df_fair['total_bytes']) * 100.0,
            50.0
        )

        for name in df_fair['scenario'].unique():
            s_data = df_fair[df_fair['scenario'] == name]
            m_share, ci_share = mean_ci(s_data['rl_share_pct'])
            print(f" Scenario: {name}")
            print(f"    RL Agent Share: {m_share:.2f}% ± {ci_share:.2f}% (95% CI)\n")
    else:
        print("No fairness data to analyze.")


def plot_aggregate_timeseries(all_timeseries_data):
    
    print(f"\n{'='*80}\n Generating Aggregate Time-Series Graphs...\n{'='*80}")
    
    if not all_timeseries_data:
        print("No time-series data to plot.")
        return

    # FIX: Filter out metadata before creating DataFrame
    timeseries_only = [d for d in all_timeseries_data if not d.get('is_metadata', False)]
    df = pd.DataFrame(timeseries_only)
    
    # FIX #2: Compute min_steps PER SCENARIO, not globally
    df['step'] = df.groupby(['scenario', 'agent_type', 'run']).cumcount()
    
    # Find min steps for each scenario separately
    min_steps_per_scenario = df.groupby(['scenario', 'agent_type'])['step'].max().groupby(level=0).min()
    
    # Filter each scenario to its own min_steps
    df = df[df.apply(lambda row: row['step'] <= min_steps_per_scenario[row['scenario']], axis=1)]

    scenarios = df['scenario'].unique()
    agents = df['agent_type'].unique()
    
    SMOOTHING_WINDOW = 20

    for scenario_name in SCENARIOS_TO_PLOT:
        if scenario_name not in scenarios:
            continue
            
        print(f"  Plotting graph for: {scenario_name}")
        scenario_data = df[df['scenario'] == scenario_name]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'Aggregate Time-Series: {scenario_name}\n(Mean of {NUM_RUNS} runs, smoothed)', fontsize=16, y=1.02)
        
        for agent in agents:
            agent_data = scenario_data[scenario_data['agent_type'] == agent]
            if agent_data.empty:
                continue

            mean_tput = agent_data.groupby('step')['tput'].mean()
            mean_rtt = agent_data.groupby('step')['rtt'].mean()
            mean_loss = agent_data.groupby('step')['loss_rate'].mean()
            
            smooth_tput = mean_tput.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            smooth_rtt = mean_rtt.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            smooth_loss = mean_loss.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            
            color = AGENT_COLORS.get(agent, '#999999')
            style = AGENT_LINESTYLES.get(agent, '-')
            
            ax1.plot(smooth_tput, label=agent, color=color, linestyle=style, linewidth=2.5)
            ax2.plot(smooth_rtt * 1000.0, label=agent, color=color, linestyle=style, linewidth=2.5)
            ax3.plot(smooth_loss, label=agent, color=color, linestyle=style, linewidth=2.5)
        
        ax1.set_ylabel('Throughput (kbps)', fontsize=12)
        ax1.set_title('Average Throughput (Smoothed)', fontsize=14)
        ax1.grid(linestyle='--', alpha=0.7)
        ax1.legend()
        
        ax2.set_ylabel('RTT (ms)', fontsize=12)
        ax2.set_title('Average RTT (Smoothed)', fontsize=14)
        ax2.grid(linestyle='--', alpha=0.7)
        ax2.legend()
        
        ax3.set_ylabel('Packet Loss Rate (%)', fontsize=12)
        ax3.set_title('Average Packet Loss (Smoothed)', fontsize=14)
        ax3.set_xlabel(f"Time ({RL_STEP_INTERVAL_MS}ms Steps)", fontsize=12)
        ax3.grid(linestyle='--', alpha=0.7)
        ax3.legend()

        plt.tight_layout()
        safe_name = re.sub(r'[^a-zA-Z0-9_]+', '', scenario_name.replace(" ", "_"))
        save_path = f"{BASE_DIR}/AGGREGATE_timeseries_{safe_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  Saved plot to {save_path}")


def plot_parameter_dynamics(all_timeseries_data):
    """Plot beta, C, and cwnd adaptation for RL agent"""
    
    print(f"\n{'='*80}\n Generating Parameter Adaptation Plots (β, C, cwnd)...\n{'='*80}")
    
    if not all_timeseries_data:
        print("No data to plot parameters.")
        return
    
    metadata_entries = [d for d in all_timeseries_data if d.get('is_metadata', False)]
    timeseries_entries = [d for d in all_timeseries_data if not d.get('is_metadata', False)]
    
    action_logs_by_scenario = {}
    for entry in metadata_entries:
        scenario = entry['scenario']
        run = entry['run']
        if scenario not in action_logs_by_scenario:
            action_logs_by_scenario[scenario] = {}
        action_logs_by_scenario[scenario][run] = entry['action_log']
    
    df = pd.DataFrame(timeseries_entries)
    
    SMOOTHING_WINDOW = 20
    
    for scenario_name in SCENARIOS_TO_PLOT:
        if scenario_name not in action_logs_by_scenario:
            print(f"  Skipping {scenario_name} (no action log data)")
            continue
        
        print(f"  Plotting parameter dynamics for: {scenario_name}")
        
        all_betas = []
        all_cs = []
        
        for run_idx, action_log in action_logs_by_scenario[scenario_name].items():
            real_betas, real_cs = get_real_params_from_actions(action_log)
            all_betas.append(real_betas)
            all_cs.append(real_cs)
        
        min_len = min(len(b) for b in all_betas)
        all_betas_trimmed = [b[:min_len] for b in all_betas]
        all_cs_trimmed = [c[:min_len] for c in all_cs]
        
        mean_beta = np.mean(all_betas_trimmed, axis=0)
        mean_c = np.mean(all_cs_trimmed, axis=0)
        std_beta = np.std(all_betas_trimmed, axis=0)
        std_c = np.std(all_cs_trimmed, axis=0)
        
        scenario_cwnd = df[(df['scenario'] == scenario_name) & (df['agent_type'] == 'RL')]
        mean_cwnd_rl = scenario_cwnd.groupby('step')['cwnd'].mean()
        
        scenario_cwnd_baseline = df[(df['scenario'] == scenario_name) & (df['agent_type'] == 'Baseline_CUBIC')]
        mean_cwnd_baseline = scenario_cwnd_baseline.groupby('step')['cwnd'].mean() if not scenario_cwnd_baseline.empty else None
        
        smooth_beta = pd.Series(mean_beta).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        smooth_c = pd.Series(mean_c).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        smooth_cwnd_rl = mean_cwnd_rl.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        smooth_cwnd_baseline = mean_cwnd_baseline.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean() if mean_cwnd_baseline is not None else None
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'Parameter Adaptation: {scenario_name}\n(Mean of {NUM_RUNS} runs, smoothed)', 
                     fontsize=16, y=1.02)
        
        ax1.plot(smooth_beta, label='RL Agent β', color='#0072B2', linewidth=2.5)
        ax1.fill_between(range(len(smooth_beta)), 
                         smooth_beta - std_beta, 
                         smooth_beta + std_beta, 
                         alpha=0.2, color='#0072B2')
        ax1.axhline(y=0.7, color='#D55E00', linestyle='--', linewidth=2, label='CUBIC Default (β=0.7)')
        ax1.set_ylabel('β (Multiplicative Decrease)', fontsize=12)
        ax1.set_title('Beta Parameter Adaptation', fontsize=14)
        ax1.set_ylim([0.3, 1.0])
        ax1.grid(linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        ax2.plot(smooth_c, label='RL Agent C', color='#0072B2', linewidth=2.5)
        ax2.fill_between(range(len(smooth_c)), 
                         smooth_c - std_c, 
                         smooth_c + std_c, 
                         alpha=0.2, color='#0072B2')
        ax2.axhline(y=0.4, color='#D55E00', linestyle='--', linewidth=2, label='CUBIC Default (C=0.4)')
        ax2.set_ylabel('C (Cubic Gain Constant)', fontsize=12)
        ax2.set_title('C Parameter Adaptation', fontsize=14)
        ax2.set_ylim([0.0, 0.8])
        ax2.grid(linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        
        ax3.plot(smooth_cwnd_rl, label='RL Agent', color='#0072B2', linewidth=2.5)
        if smooth_cwnd_baseline is not None:
            ax3.plot(smooth_cwnd_baseline, label='CUBIC Baseline', color='#D55E00', linestyle='-', linewidth=2.5)
        ax3.set_ylabel('Congestion Window (packets)', fontsize=12)
        ax3.set_xlabel(f'Time ({RL_STEP_INTERVAL_MS}ms Steps)', fontsize=12)
        ax3.set_title('Congestion Window Dynamics', fontsize=14)
        ax3.grid(linestyle='--', alpha=0.7)
        ax3.legend(loc='best')
        
        plt.tight_layout()
        safe_name = re.sub(r'[^a-zA-Z0-9_]+', '', scenario_name.replace(" ", "_"))
        save_path = f"{BASE_DIR}/PARAMS_dynamics_{safe_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  Saved parameter plot to {save_path}")


# FIX #9: Add fairness bar plot
def plot_fairness_results(all_fairness_results):
    """Generate bar chart of RL vs CUBIC bandwidth share"""
    
    print(f"\n{'='*80}\n Generating Fairness Bar Chart...\n{'='*80}")
    
    if not all_fairness_results:
        print("No fairness data to plot.")
        return
    
    df_fair = pd.DataFrame(all_fairness_results)
    df_fair['total_bytes'] = df_fair['rl_agent_bytes'] + df_fair['competing_cubic_bytes']
    df_fair['rl_share_pct'] = np.where(
        df_fair['total_bytes'] > 0,
        (df_fair['rl_agent_bytes'] / df_fair['total_bytes']) * 100.0,
        50.0
    )
    df_fair['cubic_share_pct'] = 100.0 - df_fair['rl_share_pct']
    
    scenarios = []
    rl_shares = []
    rl_cis = []
    cubic_shares = []
    cubic_cis = []
    
    for name in df_fair['scenario'].unique():
        s_data = df_fair[df_fair['scenario'] == name]
        m_rl, ci_rl = mean_ci(s_data['rl_share_pct'])
        m_cubic, ci_cubic = mean_ci(s_data['cubic_share_pct'])
        scenarios.append(name)
        rl_shares.append(m_rl)
        rl_cis.append(ci_rl)
        cubic_shares.append(m_cubic)
        cubic_cis.append(ci_cubic)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    # Side-by-side bars
    ax.bar(x_pos - width/2, rl_shares, width, yerr=rl_cis, capsize=5, 
           color='#0072B2', alpha=0.8, label='RL Agent')
    ax.bar(x_pos + width/2, cubic_shares, width, yerr=cubic_cis, capsize=5, 
           color='#D55E00', alpha=0.8, label='CUBIC Competitor')
    
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, 
               label='Ideal Fair Share (50%)', alpha=0.7)
    
    ax.set_ylabel('Bandwidth Share (%)', fontsize=12)
    ax.set_xlabel('Network Scenario', fontsize=12)
    ax.set_title(f'Fairness: RL Agent vs Competing CUBIC Flow\n(Mean ± 95% CI over {NUM_RUNS} runs)', 
                 fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace(' ', '\n') for s in scenarios], rotation=0, ha='center')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = f"{BASE_DIR}/FAIRNESS_bar_chart.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved fairness plot to {save_path}")



def calculate_loss_rate(tput_kbps, loss_count):
    bytes_recvd = (tput_kbps * 1000.0 / 8.0) * (RL_STEP_INTERVAL_MS / 1000.0)
    pkts_recvd = bytes_recvd / PKT_SIZE_BYTES
    total_pkts = pkts_recvd + loss_count
    
    loss_rate = 0.0
    if total_pkts >= 10:
        loss_rate = loss_count / total_pkts if total_pkts > 0 else 0.0
    
    return float(np.clip(loss_rate, 0.0, 1.0)) * 100.0


def main():
    model_path = MODEL_SAVE_PATH + ".zip"
    if not os.path.exists(model_path):
        return print(f" No model found at {model_path}")
    print(f" Loading trained simple model from {model_path}")

    model_env = TestNs3SimpleEnv()
    model = PPO.load(model_path, env=model_env)
    model_env.close()
    
    all_results = []
    all_fairness_results = []
    all_timeseries_data = []
    scenario_durations = {}

    print(f"\n Starting Testing ({len(TEST_SCENARIOS)} scenarios, {NUM_RUNS} runs each)\n" + "="*80)
    
    for run in range(NUM_RUNS):
        print(f"\n{'='*30} RUN {run + 1} / {NUM_RUNS} {'='*30}\n")
        base_seed = 1234 
        current_run_seed = base_seed + run
    
        for scenario_orig in TEST_SCENARIOS:
            scenario = scenario_orig.copy()
            
            print(f"\n{'='*60}\nTesting Scenario: {scenario['name']} (Run {run+1})\n{'='*60}")
            
            SHOULD_PLOT = scenario['name'] in SCENARIOS_TO_PLOT
            
            current_test_duration = TEST_DURATION_S
            scenario_durations[scenario['name']] = current_test_duration
            
            print(f"\n--- RL Test (Run {run+1}) ---")
            env = TestNs3SimpleEnv()
            try:
                env.TEST_DURATION_S = current_test_duration 
                obs, _ = env.reset(scenario=scenario, seed=current_run_seed) 
                
                total_rtt, total_loss, i_steps = 0.0, 0.0, 0
                env.debug_mode = False
                
                expected_steps = int(current_test_duration * 1000 / RL_STEP_INTERVAL_MS)
                
                for i in tqdm(range(expected_steps), desc="RL Agent Progress"):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, _, info = env.step(action)

                    if SHOULD_PLOT:
                        all_timeseries_data.append({
                            "scenario": scenario['name'],
                            "agent_type": "RL",
                            "run": run,
                            "step": i,
                            "tput": info.get('throughput', 0.0),
                            "rtt": info.get('rtt', 0.0),
                            "loss_rate": calculate_loss_rate(info.get('throughput', 0.0), info.get('loss_count', 0.0)),
                            "cwnd": info.get('cwnd', 0.0)
                        })
                    
                    if terminated:
                        print(" Episode terminated early")
                        break
                    
                    total_rtt += info.get('rtt', 0.0)
                    total_loss += info.get('loss_count', 0.0)
                    i_steps += 1
                
                # Store action log for parameter plotting
                if SHOULD_PLOT:
                    all_timeseries_data.append({
                        "scenario": scenario['name'],
                        "agent_type": "RL",
                        "run": run,
                        "action_log": env.action_log.copy(),
                        "is_metadata": True
                    })
                    
                final_bytes, _ = env.get_final_stats() 
                avg_rtt_s = (total_rtt / i_steps) if i_steps > 0 else -1.0
                
                all_output = "\n".join(env.ns3_stdout_lines + env.ns3_stderr_lines)
                total_loss_match = re.search(r"Total Packet Loss \(RL Flow\) = (\d+)", all_output)
                if not total_loss_match:
                    total_loss_match = re.search(r"Total Packet Loss\s*=\s*(\d+)", all_output, re.IGNORECASE)
                    
                final_total_loss = int(total_loss_match.group(1)) if total_loss_match else -1
                
                all_results.append({
                    "scenario": scenario['name'], 
                    "agent_type": "RL", 
                    "total_bytes": final_bytes, 
                    "avg_rtt_s": avg_rtt_s, 
                    "total_loss": final_total_loss,
                    "run": run 
                })
                
                print(f" RL Complete: {final_bytes:,} bytes, {avg_rtt_s*1000:.2f}ms RTT, {final_total_loss} loss\n")
                
            except Exception as e:
                print(f" RL Test Failed: {e}")
                import traceback; traceback.print_exc()
                all_results.append({
                    "scenario": scenario['name'], 
                    "agent_type": "RL", 
                    "total_bytes": -1, 
                    "avg_rtt_s": -1, 
                    "total_loss": -1,
                    "run": run
                })
            finally:
                env.close()

            time.sleep(2) 
            
            for baseline in BASELINE_AGENTS:
                baseline_name = baseline["name"]
                baseline_prot = baseline["tcp_prot"]
                
                print(f"\n--- Baseline Test: {baseline_name} (Run {run+1}) ---")
                
                scenario_with_seed = scenario.copy()
                scenario_with_seed['seed'] = current_run_seed

                res_bs, df_bs_data = run_baseline_test(
                    scenario_with_seed, 
                    NS3_DIR, 
                    NS3_SCRIPT, 
                    current_test_duration,
                    baseline_prot
                )
                
                all_results.append({
                    "scenario": scenario['name'], 
                    "agent_type": f"Baseline_{baseline_name}", 
                    "run": run,
                    **res_bs
                })
                
                if SHOULD_PLOT and not df_bs_data.empty:
                    for i, row in df_bs_data.iterrows():
                        all_timeseries_data.append({
                            "scenario": scenario['name'],
                            "agent_type": f"Baseline_{baseline_name}",
                            "run": run,
                            "step": i,
                            "tput": row['tput'],
                            "rtt": row['rtt'],
                            "loss_rate": calculate_loss_rate(row['tput'], row['loss_count']),
                            "cwnd": row['cwnd']
                        })

            time.sleep(2) 
            
            print(f"\n--- RL Fairness Test (Run {run+1}) ---")
            fairness_env = None
            try:
                fairness_env = TestNs3SimpleEnv(fairness_mode=True)
                fairness_env.TEST_DURATION_S = current_test_duration
                
                obs, _ = fairness_env.reset(scenario=scenario, seed=current_run_seed)
                
                fairness_env.debug_mode = False
                
                expected_steps = int(current_test_duration * 1000 / RL_STEP_INTERVAL_MS)
                
                for i in tqdm(range(expected_steps), desc="Fairness Progress"):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, _, info = fairness_env.step(action)
                    if terminated:
                        print(" Episode terminated early")
                        break
                
                rl_bytes, comp_bytes = fairness_env.get_final_stats()
                
                all_fairness_results.append({
                    "scenario": scenario['name'],
                    "rl_agent_bytes": rl_bytes,
                    "competing_cubic_bytes": comp_bytes,
                    "run": run
                })
                
            except Exception as e:
                print(f" Fairness Test Failed: {e}")
                import traceback; traceback.print_exc()
                all_fairness_results.append({
                    "scenario": scenario['name'],
                    "rl_agent_bytes": -1,
                    "competing_cubic_bytes": -1,
                    "run": run
                })
            finally:
                if fairness_env:
                    fairness_env.close()

    analyze_aggregate_results(all_results, all_fairness_results, scenario_durations)
    
    plot_aggregate_timeseries(all_timeseries_data)
    
    plot_parameter_dynamics(all_timeseries_data)
    
    plot_fairness_results(all_fairness_results)  # NEW
    
    os.makedirs(BASE_DIR, exist_ok=True) 
    
    results_df = pd.DataFrame(all_results)
    results_filename = f"{BASE_DIR}/test_results_ALL_RUNS.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\n Raw aggregate results saved to: {results_filename}")
    
    fairness_df = pd.DataFrame(all_fairness_results)
    fairness_filename = f"{BASE_DIR}/test_results_FAIRNESS_ALL_RUNS.csv"
    fairness_df.to_csv(fairness_filename, index=False)
    print(f" Raw fairness results saved to: {fairness_filename}")

    print("\n✅ Testing completed successfully!")


if __name__ == '__main__':
    try:
        import scipy
    except ImportError:
        print("\n" + "="*80)
        print("ERROR: This script requires the 'scipy' library for statistical analysis.")
        print("Please install it by running: pip install scipy")
        print("="*80 + "\n")
        exit(1)
        
    main()