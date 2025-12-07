# --- Filesystem Paths ---
NS3_DIR = "/mnt/c/BTP/simulator/ns-allinone-3.38/ns-3.38/"
NS3_SCRIPT = "congestion_simulations/main"

# --- Socket Communication Settings ---
RL_PORT = 9998
CONNECTION_TIMEOUT = 180
RL_STEP_INTERVAL_MS = 100.0 

# --- DRL Agent and Training Parameters ---
AGENT_TYPE = "PPO"
POLICY_TYPE = "MlpPolicy"
TOTAL_TIMESTEPS = 1000000 

# --- PATCH: Restore original high learning rate ---
LEARNING_RATE = 0.0003
# --- END PATCH ---

# --- Action and State Space ---
STATE_SIZE = 6 
ACTION_SIZE = 2

# --- Directories ---
BASE_DIR = "./testing_logs"
LOG_DIR = f"{BASE_DIR}/logs"
MODEL_SAVE_PATH = f"{BASE_DIR}/models/tcp_simple_rl_agent"