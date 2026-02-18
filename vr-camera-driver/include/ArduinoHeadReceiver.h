#pragma once

#include "PoseEstimator.h"
#include <atomic>
#include <mutex>
#include <thread>

class ArduinoHeadReceiver
{
public:
    explicit ArduinoHeadReceiver(int port = 4242);
    ~ArduinoHeadReceiver();

    bool start();
    void stop();

    Pose6DoF getLatestPose();

private:
    void receiveLoop();
    bool parsePacketBinary(const char *buffer, int received, Pose6DoF &outPose);

    int port_;
    std::atomic<bool> running_;
    std::thread receiverThread_;
    std::mutex poseMutex_;
    Pose6DoF latestPose_;
};
