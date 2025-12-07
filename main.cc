#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/packet-sink.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/bulk-send-helper.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("TcpRlHybrid");

struct SimState {
    double rtt{0.1};
    uint32_t cwnd{0};
    uint32_t loss_in_interval{0};
};

static SimState g_state;
static std::mutex g_state_mutex;
static int rl_socket = -1;
static std::atomic<bool> rl_connected{false};
static bool enable_rl_control = false;
static Ptr<PacketSink> global_sink = nullptr;
static double rl_step_interval_ms = 500.0;
static Ptr<FlowMonitor> g_flow_monitor = nullptr;
static uint64_t g_last_total_lost = 0;

static uint64_t g_baseline_last_bytes = 0;
static Time g_baseline_last_time = Seconds(0);
static bool g_baseline_first_call = true;
static double g_baseline_total_rtt = 0;
static double g_baseline_total_throughput = 0;
static uint32_t g_baseline_total_loss_packets = 0;
static int g_baseline_call_count = 0;

// --- Trace Callbacks ---

static void CwndTracer(uint32_t oldval, uint32_t newval) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_state.cwnd = newval;
}

static void RttTracer(Time oldval, Time newval) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_state.rtt = newval.GetSeconds();
}

// --- RL Communication ---
bool SetupRLCommunication(int port) {
    if (!enable_rl_control) return false;

    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) return false;
    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0 ||
        listen(server_socket, 1) < 0) {
        close(server_socket);
        return false;
    }
    std::cout << "NS3READYFORCONNECTION" << std::endl;
    std::cout.flush();

    std::thread accept_thread([server_socket]() {
        rl_socket = accept(server_socket, NULL, NULL);
        if (rl_socket >= 0) {
            std::cout << ">> Python RL agent connected!" << std::endl;
            rl_connected = true;
        }
        close(server_socket);
    });
    accept_thread.detach();

    return true;
}

void SendStateToRL(double throughput, uint32_t loss, double rtt, uint32_t cwnd) {
    if (!rl_connected) return;
    std::ostringstream state_msg;
    state_msg << std::fixed << std::setprecision(3)
              << throughput << "," << loss << "," << rtt << "," << cwnd << "\n";
    std::string msg = state_msg.str();
    if (send(rl_socket, msg.c_str(), msg.length(), MSG_NOSIGNAL) < 0) {
        rl_connected = false;
    }
}

std::pair<double, double> ReceiveActionFromRL() {
    if (!rl_connected) return {0.0, 0.0};
    char buffer[256];
    ssize_t bytes_received = recv(rl_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received > 0) {
        buffer[bytes_received] = '\0';
        std::string data(buffer);
        std::stringstream ss(data);
        std::string action1_str, action2_str;
        if (std::getline(ss, action1_str, ',') && std::getline(ss, action2_str)) {
            try {
                return {std::stod(action1_str), std::stod(action2_str)};
            } catch (...) {}
        }
    } else {
        rl_connected = false;
    }
    return {0.0, 0.0};
}

// TCP Cubic's default
static double g_current_beta = 0.7; 
static double g_current_c = 0.4;  


void ApplyRLAction(std::pair<double, double> actions, Ptr<Node> sourceNode) {
    if (!rl_connected) return;

    double beta_action = std::max(-1.0, std::min(1.0, actions.first));
    double c_action = std::max(-1.0, std::min(1.0, actions.second));

    const double BETA_MIN = 0.41;
    const double BETA_DEFAULT = 0.7;
    const double BETA_MAX = 0.99;
    if (beta_action >= 0) {
        g_current_beta = BETA_DEFAULT + beta_action * (BETA_MAX - BETA_DEFAULT);
    } else {
        g_current_beta = BETA_DEFAULT + beta_action * (BETA_DEFAULT - BETA_MIN);
    }

    const double C_MIN = 0.01;
    const double C_DEFAULT = 0.4;
    const double C_MAX = 0.79;
     if (c_action >= 0) {
        g_current_c = C_DEFAULT + c_action * (C_MAX - C_DEFAULT);
    } else {
        g_current_c = C_DEFAULT + c_action * (C_DEFAULT - C_MIN);
    }

    g_current_beta = std::max(BETA_MIN, std::min(BETA_MAX, g_current_beta));
    g_current_c = std::max(C_MIN, std::min(C_MAX, g_current_c));
    
    std::string tcp_path = "/NodeList/" + std::to_string(sourceNode->GetId()) +
                           "/$ns3::TcpL4Protocol/SocketList/0/$ns3::TcpSocketBase/CongestionOps/$ns3::TcpCubic";

    Config::Set(tcp_path + "/BetaCubic", DoubleValue(g_current_beta));
    Config::Set(tcp_path + "/C", DoubleValue(g_current_c));
}

// --- FlowMonitor Loss Counting ---
static void UpdateLossFromFlowMonitor() {
    if (g_flow_monitor == nullptr) return;
    g_flow_monitor->CheckForLostPackets();
    
    if (g_flow_monitor->GetFlowStats().count(1)) {
        uint64_t total_lost = g_flow_monitor->GetFlowStats().at(1).lostPackets;
        
        uint64_t interval_lost = (total_lost >= g_last_total_lost) ? (total_lost - g_last_total_lost) : 0;
        g_last_total_lost = total_lost;

        if (interval_lost > 0) {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.loss_in_interval += std::min(static_cast<uint32_t>(interval_lost), 1000u);
        }
    }
}

void PeriodicRLInteraction(Ptr<Node> sourceNode) {
    if (!enable_rl_control || Simulator::IsFinished()) return;

    static uint64_t last_bytes = 0;
    static Time last_time = Seconds(0);
    static bool first_call = true;

    Time current_time = Simulator::Now();
    double throughput = 0.0;
    
    if (g_flow_monitor != nullptr && g_flow_monitor->GetFlowStats().count(1))
    {
        FlowMonitor::FlowStats stats = g_flow_monitor->GetFlowStats().at(1);
        if (!first_call)
        {
            double time_diff = (current_time - last_time).GetSeconds();
            if (time_diff > 0)
            {
                uint64_t interval_bytes = (stats.rxBytes >= last_bytes) ? (stats.rxBytes - last_bytes) : 0;
                throughput = (interval_bytes * 8.0) / time_diff / 1000.0;
            }
        }
        else
        {
            first_call = false;
        }
        last_bytes = stats.rxBytes;
    }
    
    last_time = current_time;
    UpdateLossFromFlowMonitor();

    SimState current_state_snapshot;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        current_state_snapshot = g_state;
        g_state.loss_in_interval = 0;
    }

    if (rl_connected) {
        SendStateToRL(throughput, current_state_snapshot.loss_in_interval,
                      current_state_snapshot.rtt, current_state_snapshot.cwnd);

        std::pair<double, double> actions = ReceiveActionFromRL();
        ApplyRLAction(actions, sourceNode);
    }

    Simulator::Schedule(MilliSeconds(rl_step_interval_ms), &PeriodicRLInteraction, sourceNode);
}

void CheckForConnectionAndStartRL(Ptr<Node> sourceNode) {
    if (rl_connected) {
        std::cout << "Python agent is connected. Starting the RL control loop.\n";
        PeriodicRLInteraction(sourceNode);
    } else {
        std::cout << "Waiting for Python agent to connect...\n";
        Simulator::Schedule(MilliSeconds(100), &CheckForConnectionAndStartRL, sourceNode);
    }
}

void SetupTracing(Ptr<Node> sourceNode) {
    std::string CwndTracePath = "/NodeList/" + std::to_string(sourceNode->GetId()) +
                                "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow";
    std::string RttTracePath = "/NodeList/" + std::to_string(sourceNode->GetId()) +
                               "/$ns3::TcpL4Protocol/SocketList/0/RTT";

    Config::ConnectWithoutContext(CwndTracePath, MakeCallback(&CwndTracer));
    Config::ConnectWithoutContext(RttTracePath, MakeCallback(&RttTracer));
}

// --- Baseline Logging ---
void LogBaselineMetrics() {
    if (Simulator::IsFinished()) return;

    g_baseline_call_count++;
    Time current_time = Simulator::Now();
    double throughput = 0.0;
    
    if (g_flow_monitor != nullptr && g_flow_monitor->GetFlowStats().count(1))
    {
        FlowMonitor::FlowStats stats = g_flow_monitor->GetFlowStats().at(1);
        if (!g_baseline_first_call)
        {
            double time_diff = (current_time - g_baseline_last_time).GetSeconds();
            if (time_diff > 0)
            {
                uint64_t interval_bytes = (stats.rxBytes >= g_baseline_last_bytes) ? (stats.rxBytes - g_baseline_last_bytes) : 0;
                throughput = (interval_bytes * 8.0) / time_diff / 1000.0;
            }
        }
        else
        {
            g_baseline_first_call = false;
        }
        g_baseline_last_bytes = stats.rxBytes;
    }
    
    g_baseline_last_time = current_time;
    UpdateLossFromFlowMonitor();

    SimState current_state_snapshot;
    uint32_t lossthisinterval = 0;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        current_state_snapshot = g_state;
        lossthisinterval = g_state.loss_in_interval;
        g_state.loss_in_interval = 0;
    }

    if (g_baseline_call_count > 1) {
        g_baseline_total_rtt += current_state_snapshot.rtt;
        g_baseline_total_throughput += throughput;
        g_baseline_total_loss_packets += lossthisinterval;
    }
    
    if (g_baseline_call_count > 1) { 
        std::cout << "BASELINE_DATA," 
                  << throughput << "," 
                  << lossthisinterval << "," 
                  << current_state_snapshot.rtt << "," 
                  << current_state_snapshot.cwnd << "\n";
    }

    Simulator::Schedule(MilliSeconds(rl_step_interval_ms), &LogBaselineMetrics);
}

int main(int argc, char *argv[]) {
    std::string transport_prot = "ns3::TcpCubic";
    double error_p = 0.0;
    std::string bandwidth = "10Mbps";
    std::string delay = "20ms";
    std::string queue_size = "100p";
    double duration = 30.0;
    int rl_port = 9998; 
    
    std::string baseline_tcp_prot = "ns3::TcpCubic";
    bool enable_competing_flow = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("error_p", "Packet error rate", error_p);
    cmd.AddValue("bandwidth", "Bottleneck bandwidth", bandwidth);
    cmd.AddValue("delay", "Bottleneck delay", delay);
    cmd.AddValue("duration", "Duration", duration);
    cmd.AddValue("enable_rl", "Enable RL", enable_rl_control);
    cmd.AddValue("rl_step_interval", "RL agent interaction interval in ms", rl_step_interval_ms);
    cmd.AddValue("queue_size", "Queue size", queue_size);
    cmd.AddValue("rl_port", "Port for RL communication", rl_port);
    
    cmd.AddValue("baseline_tcp", "TCP protocol for non-RL baseline run", baseline_tcp_prot);
    cmd.AddValue("competing_flow", "Enable a competing CUBIC flow for fairness test", enable_competing_flow);
    
    cmd.Parse(argc, argv);

    if (enable_rl_control)
    {
        transport_prot = "ns3::TcpCubic";
        std::cout << "--- RL Hybrid Mode Enabled (Forcing TcpCubic) ---" << std::endl;
    }
    else
    {
        transport_prot = baseline_tcp_prot;
        std::cout << "--- Baseline Mode Enabled (Using " << transport_prot << ") ---" << std::endl;
    }

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue(transport_prot));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 21));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 21));

    NodeContainer nodes;
    nodes.Create(2);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(bandwidth));
    p2p.SetChannelAttribute("Delay", StringValue(delay));
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue(queue_size));
    NetDeviceContainer devices = p2p.Install(nodes);

    Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
    em->SetAttribute("ErrorRate", DoubleValue(error_p));
    em->SetAttribute("ErrorUnit", EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
    devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em));

    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    FlowMonitorHelper flowmonHelper;
    g_flow_monitor = flowmonHelper.InstallAll();

    // --- Setting up sink for Flow 1 (RL or Baseline) ---
    uint16_t port = 50000;
    PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = sinkHelper.Install(nodes.Get(1));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(duration + 1.0));
    global_sink = DynamicCast<PacketSink>(sinkApps.Get(0));

    // --- Setting up source for Flow 1 (RL or Baseline) ---
    double start_time = 1.0;
    BulkSendHelper sourceHelper("ns3::TcpSocketFactory", InetSocketAddress(interfaces.GetAddress(1), port));
    sourceHelper.SetAttribute("MaxBytes", UintegerValue(0));
    ApplicationContainer sourceApps = sourceHelper.Install(nodes.Get(0));
    sourceApps.Start(Seconds(start_time));
    sourceApps.Stop(Seconds(duration));

    if (enable_competing_flow)
    {
        std::cout << "--- Enabling 1 competing CUBIC flow ---" << std::endl;
        uint16_t competing_port = 50001;
        
        std::string tcp_compete_prot = "ns3::TcpCubic"; 
        
        // Competing Sink
        PacketSinkHelper competingSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), competing_port));
        ApplicationContainer competingSinkApps = competingSinkHelper.Install(nodes.Get(1));
        competingSinkApps.Start(Seconds(start_time));
        competingSinkApps.Stop(Seconds(duration + 1.0));

        // Competing Source
        BulkSendHelper competingSourceHelper("ns3::TcpSocketFactory", InetSocketAddress(interfaces.GetAddress(1), competing_port));
        competingSourceHelper.SetAttribute("MaxBytes", UintegerValue(0));
        ApplicationContainer competingSourceApps = competingSourceHelper.Install(nodes.Get(0));
        competingSourceApps.Start(Seconds(start_time));
        competingSourceApps.Stop(Seconds(duration));
    }

    Ptr<Node> sourceNode = nodes.Get(0);
    Simulator::Schedule(Seconds(start_time + 0.1), MakeBoundCallback(&SetupTracing, sourceNode));

    if (enable_rl_control) {
        if (!SetupRLCommunication(rl_port)) {
            std::cerr << "FATAL: Failed to set up RL socket. Aborting." << std::endl;
            return 1;
        }
        Simulator::Schedule(Seconds(start_time + 2.0), MakeBoundCallback(&CheckForConnectionAndStartRL, sourceNode));
    } else {
        g_baseline_last_time = Seconds(start_time + 2.0);
        Simulator::Schedule(Seconds(start_time + 2.0), &LogBaselineMetrics);
    }
    
    sourceApps.Stop(Seconds(duration + start_time)); 

    if (enable_rl_control)
    {
        Simulator::Stop(Seconds(duration + start_time + 5.0));
    }
    else
    {
        Simulator::Stop(Seconds(duration + start_time + 1.0));
    }
    
    Simulator::Run();

    std::cout.flush();
    std::cerr.flush();

    g_flow_monitor->CheckForLostPackets();
    std::map<FlowId, FlowMonitor::FlowStats> stats = g_flow_monitor->GetFlowStats();
    
    uint64_t total_lost_rl = 0;
    uint64_t total_rx_rl = 0;
    uint64_t total_lost_comp = 0;
    uint64_t total_rx_comp = 0;

    if (stats.count(1)) {
        total_lost_rl = stats[1].lostPackets;
        total_rx_rl = stats[1].rxBytes;
    }
    if (stats.count(2)) {
        total_lost_comp = stats[2].lostPackets;
        total_rx_comp = stats[2].rxBytes;
    }

    if (!enable_rl_control) {
        double avg_rtt = 0;
        double avg_throughput = 0;
        int valid_calls = g_baseline_call_count > 1 ? g_baseline_call_count - 1 : 1;
        
        if (valid_calls > 0) {
            avg_rtt = g_baseline_total_rtt / valid_calls;
            avg_throughput = g_baseline_total_throughput / valid_calls;
        }
       
        std::cout << "========== BASELINE FINAL METRICS ==========" << std::endl;
        std::cout << "Average RTT = " << avg_rtt << " s" << std::endl;
        std::cout << "Average Throughput = " << avg_throughput << " kbps" << std::endl;
        std::cout << "Total Packet Loss = " << total_lost_rl << std::endl;
        std::cout << "Total Rx Bytes = " << total_rx_rl << std::endl;
        
        if (enable_competing_flow) {
            std::cout << "Total Rx Bytes (Competing Flow) = " << total_rx_comp << std::endl;
            std::cout << "Total Packet Loss (Competing Flow) = " << total_lost_comp << std::endl;
        }
        std::cout << "============================================" << std::endl;
    } else {
        std::cout << "========== SIMULATION FINISHED ==========" << std::endl;
        std::cout << "Total Rx Bytes (RL Flow) = " << total_rx_rl << std::endl;
        std::cout << "Total Packet Loss (RL Flow) = " << total_lost_rl << std::endl;
        if (enable_competing_flow) {
            std::cout << "Total Rx Bytes (Competing Flow) = " << total_rx_comp << std::endl;
            std::cout << "Total Packet Loss (Competing Flow) = " << total_lost_comp << std::endl;
        }
        std::cout << "=========================================" << std::endl;
    }

    std::cout.flush();
    std::cerr.flush();

    if (rl_socket >= 0) close(rl_socket);

    Simulator::Destroy();
    return 0;
}
