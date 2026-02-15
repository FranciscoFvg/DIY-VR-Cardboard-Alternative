#include "PoseEstimator.h"
#include "PoseFilter.h"
#include "HandPoseReceiver.h"
#include "CameraVRDriver.h"
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    std::cout << "=== CameraVR - Driver de Rastreamento por Câmera ===" << std::endl;

    // 1. Inicializar receptor de pose (UDP)
    HandPoseReceiver receiver(7000);
    receiver.start();

    // 3. Inicializar estimador de pose e filtro
    PoseEstimator poseEstimator;
    PoseFilter leftFilter;
    PoseFilter rightFilter;
    leftFilter.setAlpha(0.7);  // Filtro complementar
    rightFilter.setAlpha(0.7); // Filtro complementar

    struct PoseState
    {
        Pose6DoF lastPose;
        Eigen::Vector3d velocity{0.0, 0.0, 0.0};
        double lastTime = 0.0;
        bool hasPose = false;
    };

    PoseState leftState;
    PoseState rightState;

    const double holdDuration = 0.10;
    const double predictDuration = 0.35;

    // 4. Loop principal
    auto lastTime = std::chrono::high_resolution_clock::now();

    std::cout << "\nRastreando maos via MediaPipe (UDP)..." << std::endl;
    std::cout << "Inicie o script hand_tracker.py" << std::endl;
    std::cout << "\n[DEBUG] Aguardando poses UDP..." << std::endl;

    int frameCount = 0;
    auto debugTimer = std::chrono::high_resolution_clock::now();

    while (true)
    {
        // Calcular dt
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;

        double nowSec = std::chrono::duration<double>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        Pose6DoF leftPose = receiver.getLatestPose(HandPoseReceiver::Hand::Left);
        Pose6DoF rightPose = receiver.getLatestPose(HandPoseReceiver::Hand::Right);
        float leftTrigger = receiver.getLatestTrigger(HandPoseReceiver::Hand::Left);
        float rightTrigger = receiver.getLatestTrigger(HandPoseReceiver::Hand::Right);

        // Debug: imprimir a cada 2 segundos
        frameCount++;
        if (std::chrono::duration<double>(now - debugTimer).count() > 2.0)
        {
            std::cout << "[DEBUG] Frame " << frameCount << " - Left: " 
                      << (leftPose.valid ? "VALIDA" : "invalida")
                      << " (Trig=" << leftTrigger << ")"
                      << ", Right: " << (rightPose.valid ? "VALIDA" : "invalida")
                      << " (Trig=" << rightTrigger << ")"
                      << std::endl;
            debugTimer = now;
        }

        auto synthesizePose = [&](const Pose6DoF &input, PoseState &state) -> Pose6DoF
        {
            Pose6DoF output = input;

            if (input.valid)
            {
                if (state.hasPose)
                {
                    double dtPose = input.timestamp - state.lastTime;
                    if (dtPose > 0.001 && dtPose < 1.0)
                    {
                        state.velocity = (input.position - state.lastPose.position) / dtPose;
                    }
                }

                state.lastPose = input;
                state.lastTime = input.timestamp;
                state.hasPose = true;
                return output;
            }

            if (!state.hasPose)
            {
                output.valid = false;
                return output;
            }

            double age = nowSec - state.lastTime;
            if (age <= holdDuration)
            {
                output = state.lastPose;
                output.timestamp = nowSec;
                output.valid = true;
                return output;
            }

            if (age <= predictDuration)
            {
                output = state.lastPose;
                output.position = state.lastPose.position + state.velocity * age;
                output.timestamp = nowSec;
                output.valid = true;
                return output;
            }

            output.valid = false;
            return output;
        };

        if (leftPose.valid && (nowSec - leftPose.timestamp) > 0.5)
        {
            leftPose.valid = false;
        }

        if (rightPose.valid && (nowSec - rightPose.timestamp) > 0.5)
        {
            rightPose.valid = false;
        }

        leftPose = synthesizePose(leftPose, leftState);
        rightPose = synthesizePose(rightPose, rightState);

        if (VirtualController *leftController = g_CameraVRDriver.GetController(0))
        {
            Pose6DoF pose = leftPose.valid ? poseEstimator.transformToVRSpace(leftPose) : leftPose;
            pose = leftPose.valid ? leftFilter.filterComplementary(pose, dt) : pose;
            leftController->UpdatePose(pose);
            leftController->UpdateTrigger(leftPose.valid ? leftTrigger : 0.0f);
        }

        if (VirtualController *rightController = g_CameraVRDriver.GetController(1))
        {
            Pose6DoF pose = rightPose.valid ? poseEstimator.transformToVRSpace(rightPose) : rightPose;
            pose = rightPose.valid ? rightFilter.filterComplementary(pose, dt) : pose;
            rightController->UpdatePose(pose);
            rightController->UpdateTrigger(rightPose.valid ? rightTrigger : 0.0f);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    receiver.stop();
    std::cout << "Encerrando..." << std::endl;

    return 0;
}
