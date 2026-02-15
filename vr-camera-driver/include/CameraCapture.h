#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <atomic>
#include <thread>

class CameraCapture
{
public:
    CameraCapture(int cameraIndex = 0);
    ~CameraCapture();

    bool initialize();
    void start();
    void stop();

    cv::Mat getLatestFrame();
    bool isRunning() const { return running_; }

    void setCameraIndex(int index) { cameraIndex_ = index; }
    int getCameraIndex() const { return cameraIndex_; }

private:
    void captureLoop();

    int cameraIndex_;
    cv::VideoCapture capture_;
    cv::Mat latestFrame_;
    std::mutex frameMutex_;
    std::atomic<bool> running_;
    std::thread captureThread_;
};
