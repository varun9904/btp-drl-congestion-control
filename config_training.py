# --- Filesystem Paths ---
NS3_DIR = "/mnt/c/BTP/simulator/ns-allinone-3.38/ns-3.38/"
NS3_SCRIPT = "congestion_simulations/main"

# --- Socket Communication Settings ---
RL_PORT = 9997
CONNECTION_TIMEOUT = 60
RL_STEP_INTERVAL_MS = 100.0 

# --- DRL Agent and Training Parameters ---
AGENT_TYPE = "PPO"
POLICY_TYPE = "MlpPolicy"
TOTAL_TIMESTEPS = 1000000 

LEARNING_RATE = 0.0003

# --- Action and State Space ---
STATE_SIZE = 6 
ACTION_SIZE = 2

# --- Directories ---
BASE_DIR = "./testing_logs"
LOG_DIR = f"{BASE_DIR}/logs"
MODEL_SAVE_PATH = f"{BASE_DIR}/models/tcp_simple_rl_agent"