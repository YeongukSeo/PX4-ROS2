#include <chrono>
#include <cmath>
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardFinalSquare : public rclcpp::Node {
public:
    OffboardFinalSquare() : Node("offboard_final_square") {
        auto qos_profile = rclcpp::SensorDataQoS();

        // Publishers
        offboard_control_mode_publisher_ =
            this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ =
            this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ =
            this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        // Subscribers
        odometry_subscription_ = this->create_subscription<VehicleOdometry>(
            "/fmu/out/vehicle_odometry", qos_profile,
            [this](const VehicleOdometry &msg) {
                // Position (NED)
                current_pos_ = {msg.position[0], msg.position[1], msg.position[2]};

                // Yaw (quaternion -> yaw), PX4 VehicleOdometry.q = [w, x, y, z]
                const float qw = msg.q[0];
                const float qx = msg.q[1];
                const float qy = msg.q[2];
                const float qz = msg.q[3];

                const float siny_cosp = 2.0f * (qw * qz + qx * qy);
                const float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
                current_yaw_meas_ = std::atan2(siny_cosp, cosy_cosp);

                // Capture yaw once before arming/takeoff and keep it during warmup/takeoff
                if (!yaw_initialized_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    target_yaw_ = current_yaw_sp_;
                    yaw_initialized_ = true;
                }

                // During warmup/takeoff: hold XY at current position
                if (phase_ == Phase::warmup || phase_ == Phase::takeoff) {
                    setpoint_pos_[0] = current_pos_[0];
                    setpoint_pos_[1] = current_pos_[1];
                }
            });

        status_subscription_ = this->create_subscription<VehicleStatus>(
            "/fmu/out/vehicle_status_v1", qos_profile,
            [this](const VehicleStatus &msg) {
                arming_state_ = msg.arming_state;
            });

        // Timer: 100ms (10Hz)
        timer_ = this->create_wall_timer(100ms, [this]() {
            if (phase_ == Phase::landing) {
                if (arming_state_ == VehicleStatus::ARMING_STATE_DISARMED) {
                    RCLCPP_INFO(this->get_logger(),
                                ">>> Disarm confirmed. Mission complete. Shutting down.");
                    rclcpp::shutdown();
                    return;
                }
            }

            if (phase_ != Phase::landing) {
                publish_offboard_control_mode();
                publish_trajectory_setpoint();
            }

            manage_mission_flow();
        });
    }

private:
    // =========================
    // 1) State / Parameters
    // =========================
    enum class Phase { warmup, takeoff, move_straight, yaw_turn, landing };

    // ROS interfaces
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscription_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscription_;

    // Mission phase
    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;

    // Vehicle state
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};

    // Setpoint "bait"
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f};

    // Yaw: capture-and-hold during warmup/takeoff, then turn-in-place at WP
    bool  yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f;
    float current_yaw_sp_   = -3.14f;
    float target_yaw_       = -3.14f;

    // Arming state
    uint8_t arming_state_ = 0;
    bool arm_sent_ = false;
    bool offboard_sent_ = false;

    // Timing
    const float dt_ = 0.1f;                 // 10 Hz
    const uint64_t warmup_armed_ticks_ = 50; // 5 seconds idle spin

    // Altitude
    const float flight_alt_ = 50.0f;

    // Yaw rate limit (rad/s)
    const float yaw_speed_ = 0.5f;

    // Waypoint behavior
    const float wp_switch_radius_ = 5.0f;   // reach radius (m)

    // Speed profile (smooth accel/decel)
    float speed_sp_ = 0.0f;
    const float v_max_ = 8.0f;              // max speed (m/s)
    const float a_max_ = 1.5f;              // accel limit (m/s^2)

    // Waypoints (local NED XY)
    // NOTE: Must have at least 2 points. Here last point repeats to close loop.
    std::vector<std::array<float, 2>> waypoints_ = {
        {0.0f,   0.0f},
        {100.0f, 0.0f},
        {100.0f, 100.0f},
        {0.0f,   100.0f},
        {0.0f,   0.0f}
    };
    size_t seg_index_ = 0; // segment: waypoints_[seg_index_] -> waypoints_[seg_index_+1]

    // =========================
    // 2) Utility / Math
    // =========================
    static float wrap_angle(float angle) {
        while (angle > M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }

    static float clampf(float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    }

    // =========================
    // 3) Setpoint Generation
    // =========================
    void update_yaw_smoothing() {
        const float diff = wrap_angle(target_yaw_ - current_yaw_sp_);
        const float step = yaw_speed_ * dt_;

        if (std::abs(diff) < step) {
            current_yaw_sp_ = target_yaw_;
        } else {
            current_yaw_sp_ += (diff > 0.0f) ? step : -step;
        }
        current_yaw_sp_ = wrap_angle(current_yaw_sp_);
    }

    // Speed profile based on remaining distance (decel) + accel rate limit
    void update_speed_profile(float dist_to_goal) {
        const float v_decel_limit = std::sqrt(std::max(0.0f, 2.0f * a_max_ * dist_to_goal));
        const float v_cmd = std::min(v_max_, v_decel_limit);

        const float dv_max = a_max_ * dt_;
        const float dv = clampf(v_cmd - speed_sp_, -dv_max, dv_max);
        speed_sp_ = std::max(0.0f, speed_sp_ + dv);
    }

    // =========================
    // 4) PX4 Publish Helpers
    // =========================
    void publish_offboard_control_mode() {
        OffboardControlMode msg{};
        msg.position = true;
        msg.velocity = false;
        msg.acceleration = false;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};

        msg.position[0] = setpoint_pos_[0];
        msg.position[1] = setpoint_pos_[1];

        if (phase_ == Phase::warmup) {
            // Hold ground Z during warmup so motors can idle-spin without takeoff
            msg.position[2] = current_pos_[2];
        } else {
            // Command target altitude during takeoff/move
            msg.position[2] = -flight_alt_;
        }

        msg.yaw = current_yaw_sp_;

        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_publisher_->publish(msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0f, float param2 = 0.0f) {
        VehicleCommand msg{};
        msg.param1 = param1;
        msg.param2 = param2;
        msg.command = command;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_command_publisher_->publish(msg);
    }

    // =========================
    // 5) Mission / FSM
    // =========================
    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup: {
                if (!yaw_initialized_) {
                    if (++ticks_ % 10 == 0) {
                        RCLCPP_INFO(this->get_logger(), "Waiting for odometry to capture yaw...");
                    }
                    break;
                }

                if (!offboard_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    offboard_sent_ = true;
                    RCLCPP_INFO(this->get_logger(), "Offboard mode request sent.");
                }

                if (!arm_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    arm_sent_ = true;
                    ticks_ = 0; // start counting idle warmup after arming request
                    RCLCPP_INFO(this->get_logger(),
                                "Arm request sent. Idling motors for %lu ticks (%.1f s)...",
                                warmup_armed_ticks_, warmup_armed_ticks_ * dt_);
                    break;
                }

                if (++ticks_ >= warmup_armed_ticks_) {
                    RCLCPP_INFO(this->get_logger(), "Idle warmup complete. Starting takeoff...");
                    phase_ = Phase::takeoff;
                    ticks_ = 0;
                }
            } break;

            case Phase::takeoff: {
                if (++ticks_ % 10 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Climbing... Alt: %.1fm", -current_pos_[2]);
                }

                // Keep yaw fixed during takeoff (do NOT change yaw target here)
                current_yaw_sp_ = target_yaw_;

                if (-current_pos_[2] >= (flight_alt_ - 1.0f)) {
                    RCLCPP_INFO(this->get_logger(), "Reached altitude. Start moving to WP1...");
                    seg_index_ = 0;
                    speed_sp_ = 0.0f;

                    // Start setpoint from current position to avoid jumps
                    setpoint_pos_[0] = current_pos_[0];
                    setpoint_pos_[1] = current_pos_[1];
                    setpoint_pos_[2] = -flight_alt_;

                    // Initialize target yaw toward first segment direction
                    if (seg_index_ + 1 < waypoints_.size()) {
                        const auto &A = waypoints_[seg_index_];
                        const auto &B = waypoints_[seg_index_ + 1];
                        target_yaw_ = std::atan2(B[1] - A[1], B[0] - A[0]);
                        current_yaw_sp_ = target_yaw_;
                    }

                    phase_ = Phase::move_straight;
                    ticks_ = 0;
                }
            } break;

            case Phase::move_straight: {
                // Need at least a next waypoint
                if (seg_index_ + 1 >= waypoints_.size()) {
                    RCLCPP_INFO(this->get_logger(), "All segments done. Landing...");
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
                    phase_ = Phase::landing;
                    break;
                }

                const auto &B = waypoints_[seg_index_ + 1];

                const float dx = B[0] - current_pos_[0];
                const float dy = B[1] - current_pos_[1];
                const float dist_to_B = std::sqrt(dx * dx + dy * dy);

                // Speed planning
                update_speed_profile(dist_to_B);

                // Move the setpoint forward toward B (straight)
                if (dist_to_B > 0.1f && speed_sp_ > 0.01f) {
                    const float step = std::min(speed_sp_ * dt_, dist_to_B);
                    const float ux = dx / dist_to_B;
                    const float uy = dy / dist_to_B;

                    setpoint_pos_[0] += ux * step;
                    setpoint_pos_[1] += uy * step;
                } else {
                    setpoint_pos_[0] = B[0];
                    setpoint_pos_[1] = B[1];
                }
                setpoint_pos_[2] = -flight_alt_;

                // IMPORTANT: yaw fixed during move (no continuous yaw tracking)
                current_yaw_sp_ = target_yaw_;

                // If reached waypoint, stop XY and turn-in-place to next segment direction
                if (dist_to_B < wp_switch_radius_) {
                    RCLCPP_INFO(this->get_logger(), "Reached WP %zu. Turning to next heading...", seg_index_ + 1);
                    speed_sp_ = 0.0f;

                    // If there is another waypoint after B, set yaw toward next segment
                    if (seg_index_ + 2 < waypoints_.size()) {
                        const auto &C = waypoints_[seg_index_ + 2];
                        target_yaw_ = std::atan2(C[1] - B[1], C[0] - B[0]);
                        phase_ = Phase::yaw_turn;
                    } else {
                        // No further waypoint -> land
                        RCLCPP_INFO(this->get_logger(), "Final WP reached. Landing...");
                        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
                        phase_ = Phase::landing;
                    }
                    ticks_ = 0;
                }
            } break;

            case Phase::yaw_turn: {
                // Hold XY exactly at current position while rotating yaw
                setpoint_pos_[0] = current_pos_[0];
                setpoint_pos_[1] = current_pos_[1];
                setpoint_pos_[2] = -flight_alt_;

                update_yaw_smoothing();

                // When aligned, proceed to next segment
                if (std::abs(wrap_angle(target_yaw_ - current_yaw_sp_)) < 0.05f) {
                    seg_index_++;
                    RCLCPP_INFO(this->get_logger(), "Yaw aligned. Moving to next WP %zu...", seg_index_ + 1);
                    phase_ = Phase::move_straight;
                    ticks_ = 0;
                }
            } break;

            case Phase::landing:
                break;
        }
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardFinalSquare>());
    rclcpp::shutdown();
    return 0;
}