#include "HandPoseReceiver.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

HandPoseReceiver::HandPoseReceiver(int port)
    : port_(port), running_(false) {
    leftPose_.valid = false;
    rightPose_.valid = false;
    leftTrigger_ = 0.0f;
    rightTrigger_ = 0.0f;
    cameraHeadMounted_ = false;
    followHeadTranslation_ = false;
}

HandPoseReceiver::~HandPoseReceiver() {
    stop();
}

bool HandPoseReceiver::start() {
    if (running_) {
        return true;
    }

    running_ = true;
    receiverThread_ = std::thread(&HandPoseReceiver::receiveLoop, this);
    return true;
}

void HandPoseReceiver::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    if (receiverThread_.joinable()) {
        receiverThread_.join();
    }
}

Pose6DoF HandPoseReceiver::getLatestPose(HandPoseReceiver::Hand hand) {
    std::lock_guard<std::mutex> lock(poseMutex_);
    return (hand == Hand::Left) ? leftPose_ : rightPose_;
}

float HandPoseReceiver::getLatestTrigger(HandPoseReceiver::Hand hand) {
    std::lock_guard<std::mutex> lock(poseMutex_);
    return (hand == Hand::Left) ? leftTrigger_ : rightTrigger_;
}

bool HandPoseReceiver::isCameraHeadMounted() {
    std::lock_guard<std::mutex> lock(poseMutex_);
    return cameraHeadMounted_;
}

bool HandPoseReceiver::isHeadTranslationFollowEnabled() {
    std::lock_guard<std::mutex> lock(poseMutex_);
    return followHeadTranslation_;
}

void HandPoseReceiver::updatePose(HandPoseReceiver::Hand hand, const Pose6DoF& pose, float triggerValue) {
    std::lock_guard<std::mutex> lock(poseMutex_);
    if (hand == Hand::Left) {
        leftPose_ = pose;
        leftTrigger_ = triggerValue;
    } else {
        rightPose_ = pose;
        rightTrigger_ = triggerValue;
    }
}

void HandPoseReceiver::receiveLoop() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Falha ao inicializar WinSock" << std::endl;
        return;
    }
#endif

    int sock = (int)socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Falha ao criar socket UDP" << std::endl;
        return;
    }

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)port_);

    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Falha ao bind socket UDP" << std::endl;
#ifdef _WIN32
        closesocket(sock);
        WSACleanup();
#else
        close(sock);
#endif
        return;
    }

    char buffer[512];

    while (running_) {
        sockaddr_in fromAddr;
#ifdef _WIN32
        int fromLen = sizeof(fromAddr);
        int received = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, (sockaddr*)&fromAddr, &fromLen);
#else
        socklen_t fromLen = sizeof(fromAddr);
        int received = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, (sockaddr*)&fromAddr, &fromLen);
#endif

        if (received <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        buffer[received] = '\0';
        std::istringstream iss(buffer);
        char handChar = 0;
        double px = 0, py = 0, pz = 0;
        double qw = 1, qx = 0, qy = 0, qz = 0;
        int valid = 0;
        double timestamp = 0.0;
        double triggerValue = 0.0;
        int cameraOnHead = 0;
        int followHeadTranslation = 0;

        iss >> handChar >> px >> py >> pz >> qw >> qx >> qy >> qz >> valid >> timestamp;
        if (iss.fail()) {
            continue;
        }

        if (!(iss >> triggerValue)) {
            triggerValue = 0.0;
        }

        if (!(iss >> cameraOnHead)) {
            cameraOnHead = 0;
        }

        if (!(iss >> followHeadTranslation)) {
            followHeadTranslation = 0;
        }

        Pose6DoF pose;
        pose.position = Eigen::Vector3d(px, py, pz);
        pose.rotation = Eigen::Quaterniond(qw, qx, qy, qz);
        pose.rotation.normalize();
        pose.timestamp = timestamp;
        pose.valid = (valid != 0);
        pose.matrix = Eigen::Matrix4d::Identity();
        pose.matrix.block<3, 3>(0, 0) = pose.rotation.toRotationMatrix();
        pose.matrix.block<3, 1>(0, 3) = pose.position;

        {
            std::lock_guard<std::mutex> lock(poseMutex_);
            cameraHeadMounted_ = (cameraOnHead != 0);
            followHeadTranslation_ = (followHeadTranslation != 0);
        }

        if (handChar == 'L' || handChar == 'l') {
            updatePose(Hand::Left, pose, (float)triggerValue);
        } else if (handChar == 'R' || handChar == 'r') {
            updatePose(Hand::Right, pose, (float)triggerValue);
        }
    }

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif
}
