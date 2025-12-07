import os
import re
import threading
import socket
import psutil 


from flask import Flask, request, jsonify, send_from_directory
from stable_baselines3 import PPO


# Ensure these imports match your local file structure
from testing_agent2 import (
    TestNs3SimpleEnv,
    run_baseline_test,
    calculate_loss_rate,
    RL_STEP_INTERVAL_MS,
    TEST_DURATION_S,
    BASELINE_AGENTS,
)
from config_testing import NS3_DIR, NS3_SCRIPT, MODEL_SAVE_PATH



app = Flask(__name__)


_model_lock = threading.Lock()
_model = None
_simulation_lock = threading.Lock()



def _get_model():
    global _model
    with _model_lock:
        if _model is not None:
            return _model


        env = TestNs3SimpleEnv()
        model_path = MODEL_SAVE_PATH + ".zip"
        if not os.path.exists(model_path):
            env.close()
            raise RuntimeError(f"No trained model found at {model_path}")


        _model = PPO.load(model_path, env=env)
        env.close()
        return _model



def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))



def _build_scenario(bandwidth_mbps, delay_ms, loss_percent):
    bandwidth_mbps = _clamp(float(bandwidth_mbps), 1.0, 200.0)
    delay_ms = _clamp(float(delay_ms), 5.0, 500.0)
    loss_percent = _clamp(float(loss_percent), 0.0, 4.0)


    queue_size = int(max(int((bandwidth_mbps * 1e6) * (2 * delay_ms / 1000) / (1500 * 8)), 20))
    scenario = {
        "name": "Custom Interactive Scenario",
        "bandwidth": f"{bandwidth_mbps:.1f}Mbps",
        "delay": f"{delay_ms:.1f}ms",
        "loss": loss_percent / 100.0,
        "queue_size": f"{queue_size}p",
    }


    normalized = {
        "bandwidth_mbps": bandwidth_mbps,
        "delay_ms": delay_ms,
        "loss_percent": loss_percent,
    }


    return scenario, normalized



def _run_rl_once(scenario, seed=1234):
    model = _get_model()
    env = TestNs3SimpleEnv()
    env.TEST_DURATION_S = TEST_DURATION_S


    obs, _ = env.reset(scenario=scenario, seed=seed)


    timeseries = []
    total_rtt_s = 0.0
    total_loss = 0.0
    steps = 0


    expected_steps = int(TEST_DURATION_S * 1000.0 / RL_STEP_INTERVAL_MS)


    for i in range(expected_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(action)


        tput_kbps = float(info.get("throughput", 0.0))
        rtt_s = float(info.get("rtt", 0.0))
        loss_count = float(info.get("loss_count", 0.0))


        timeseries.append({
            "step": i,
            "tput_kbps": tput_kbps,
            "rtt_ms": rtt_s * 1000.0,
            "loss_rate_pct": calculate_loss_rate(tput_kbps, loss_count),
        })


        total_rtt_s += rtt_s
        total_loss += loss_count
        steps += 1


        if terminated:
            break


    final_bytes, _ = env.get_final_stats()
    avg_rtt_ms = (total_rtt_s / steps) * 1000.0 if steps > 0 else -1.0


    all_output = "\n".join(env.ns3_stdout_lines + env.ns3_stderr_lines)
    loss_match = re.search(r"Total Packet Loss \(RL Flow\) = (\d+)", all_output)
    if not loss_match:
        loss_match = re.search(r"Total Packet Loss\s*=\s*(\d+)", all_output, re.IGNORECASE)
    final_total_loss = int(loss_match.group(1)) if loss_match else int(total_loss)


    env.close()


    duration_s = TEST_DURATION_S
    throughput_mbps = 0.0
    if final_bytes > 0 and duration_s > 0:
        throughput_mbps = (final_bytes * 8.0) / (duration_s * 1e6)


    summary = {
        "total_bytes": int(final_bytes),
        "throughput_mbps": float(throughput_mbps),
        "avg_rtt_ms": float(avg_rtt_ms),
        "total_loss": int(final_total_loss),
    }


    return summary, timeseries



def _run_cubic_once(scenario, seed=1234):
    scenario_with_seed = scenario.copy()
    scenario_with_seed["seed"] = seed


    baseline_tcp = BASELINE_AGENTS[0]["tcp_prot"] if BASELINE_AGENTS else "ns3::TcpCubic"


    summary_raw, df_bs = run_baseline_test(
        scenario_with_seed,
        NS3_DIR,
        NS3_SCRIPT,
        TEST_DURATION_S,
        baseline_tcp,
    )


    timeseries = []
    for i, row in df_bs.iterrows():
        tput_kbps = float(row.get("tput", 0.0))
        loss_count = float(row.get("loss_count", 0.0))
        rtt_s = float(row.get("rtt", 0.0))


        timeseries.append({
            "step": int(i),
            "tput_kbps": tput_kbps,
            "rtt_ms": rtt_s * 1000.0,
            "loss_rate_pct": calculate_loss_rate(tput_kbps, loss_count),
        })


    duration_s = TEST_DURATION_S
    total_bytes = int(summary_raw.get("total_bytes", -1))
    avg_rtt_ms = float(summary_raw.get("avg_rtt_s", 0.0) * 1000.0)
    total_loss = int(summary_raw.get("total_loss", 0))


    throughput_mbps = 0.0
    if total_bytes > 0 and duration_s > 0:
        throughput_mbps = (total_bytes * 8.0) / (duration_s * 1e6)


    summary = {
        "total_bytes": total_bytes,
        "throughput_mbps": float(throughput_mbps),
        "avg_rtt_ms": avg_rtt_ms,
        "total_loss": total_loss,
    }


    return summary, timeseries



@app.route("/", methods=["GET"])
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, "index.html")



@app.route("/api/run", methods=["POST"])
def api_run():
    if not _simulation_lock.acquire(blocking=False):
        return jsonify({"error": "A simulation is already running. Please wait and try again."}), 429


    try:
        data = request.get_json(force=True) or {}


        bandwidth_mbps = float(data.get("bandwidth_mbps", 40.0))
        # CHANGED: now expects delay_ms instead of rtt_ms
        delay_ms = float(data.get("delay_ms", 100.0))
        loss_percent = float(data.get("loss_percent", 1.0))


        scenario, normalized = _build_scenario(bandwidth_mbps, delay_ms, loss_percent)
        seed = int(data.get("seed", 1234))


        rl_summary, rl_ts = _run_rl_once(scenario, seed=seed)
        cubic_summary, cubic_ts = _run_cubic_once(scenario, seed=seed)


        return jsonify({
            "step_interval_ms": RL_STEP_INTERVAL_MS,
            "scenario": normalized,
            "rl": {
                "summary": rl_summary,
                "timeseries": rl_ts,
            },
            "baseline": {
                "name": "CUBIC",
                "summary": cubic_summary,
                "timeseries": cubic_ts,
            },
        })


    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


    finally:
        _simulation_lock.release()



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
