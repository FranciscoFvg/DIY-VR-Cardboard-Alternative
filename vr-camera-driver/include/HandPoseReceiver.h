#pragma once

#include "PoseEstimator.h"
#include <atomic>
#include <mutex>
#include <thread>

class HandPoseReceiver {
public:
    enum class Hand { Left, Right };

    explicit HandPoseReceiver(int port = 7000);
    ~HandPoseReceiver();

    bool start();
    void stop();

    Pose6DoF getLatestPose(Hand hand);
    float getLatestTrigger(Hand hand);
    bool isCameraHeadMounted();
    bool isHeadTranslationFollowEnabled();

private:
    void receiveLoop();
    void updatePose(Hand hand, const Pose6DoF& pose, float triggerValue);

    int port_;
    std::atomic<bool> running_;
    std::thread receiverThread_;
    std::mutex poseMutex_;
    Pose6DoF leftPose_;
    Pose6DoF rightPose_;
    float leftTrigger_;
    float rightTrigger_;
    bool cameraHeadMounted_;
    bool followHeadTranslation_;
};
