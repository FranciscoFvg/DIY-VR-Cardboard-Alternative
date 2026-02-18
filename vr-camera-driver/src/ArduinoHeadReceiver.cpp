#include "ArduinoHeadReceiver.h"

#include <Eigen/Geometry>
#include <chrono>
#include <cstring>
#include <iostream>

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

namespace
{
constexpr double kPi = 3.14159265358979323846;

Eigen::Quaterniond eulerDegToQuaternion(double yawDeg, double pitchDeg, double rollDeg)
{
    const double yawRad = yawDeg * kPi / 180.0;
    const double pitchRad = pitchDeg * kPi / 180.0;
    const double rollRad = rollDeg * kPi / 180.0;

    Eigen::AngleAxisd yawAxis(yawRad, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAxis(pitchRad, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rollAxis(rollRad, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAxis * pitchAxis * rollAxis;
    q.normalize();
    return q;
}
}

ArduinoHeadReceiver::ArduinoHeadReceiver(int port)
    : port_(port), running_(false)
{
    latestPose_.valid = false;
}

ArduinoHeadReceiver::~ArduinoHeadReceiver()
{
    stop();
}

bool ArduinoHeadReceiver::start()
{
    if (running_)
    {
        return true;
    }

    running_ = true;
    receiverThread_ = std::thread(&ArduinoHeadReceiver::receiveLoop, this);
    return true;
}

void ArduinoHeadReceiver::stop()
{
    if (!running_)
    {
        return;
    }

    running_ = false;
    if (receiverThread_.joinable())
    {
        receiverThread_.join();
    }
}

Pose6DoF ArduinoHeadReceiver::getLatestPose()
{
    std::lock_guard<std::mutex> lock(poseMutex_);
    return latestPose_;
}

bool ArduinoHeadReceiver::parsePacketBinary(const char *buffer, int received, Pose6DoF &outPose)
{
    if (received < static_cast<int>(sizeof(double) * 6))
    {
        return false;
    }

    double values[6];
    std::memcpy(values, buffer, sizeof(values));

    const double x = values[0];
    const double y = values[1];
    const double z = values[2];
    const double yaw = values[3];
    const double pitch = values[4];
    const double roll = values[5];

    Pose6DoF pose;
    pose.position = Eigen::Vector3d(x, y, z);
    pose.rotation = eulerDegToQuaternion(yaw, pitch, roll);
    pose.timestamp = std::chrono::duration<double>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    pose.valid = true;
    pose.matrix = Eigen::Matrix4d::Identity();
    pose.matrix.block<3, 3>(0, 0) = pose.rotation.toRotationMatrix();
    pose.matrix.block<3, 1>(0, 3) = pose.position;

    outPose = pose;
    return true;
}

void ArduinoHeadReceiver::receiveLoop()
{
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        std::cerr << "Falha ao inicializar WinSock (ArduinoHeadReceiver)" << std::endl;
        return;
    }
#endif

    int sock = static_cast<int>(socket(AF_INET, SOCK_DGRAM, 0));
    if (sock < 0)
    {
        std::cerr << "Falha ao criar socket UDP (ArduinoHeadReceiver)" << std::endl;
        return;
    }

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<unsigned short>(port_));

    if (bind(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        std::cerr << "Falha ao bind socket UDP na porta " << port_ << " (ArduinoHeadReceiver)" << std::endl;
#ifdef _WIN32
        closesocket(sock);
        WSACleanup();
#else
        close(sock);
#endif
        return;
    }

    std::cout << "[ArduinoHeadReceiver] Escutando UDP na porta " << port_ << std::endl;

    char buffer[256];
    while (running_)
    {
        sockaddr_in fromAddr;
#ifdef _WIN32
        int fromLen = sizeof(fromAddr);
        int received = recvfrom(sock, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr *>(&fromAddr), &fromLen);
#else
        socklen_t fromLen = sizeof(fromAddr);
        int received = recvfrom(sock, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr *>(&fromAddr), &fromLen);
#endif

        if (received <= 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        Pose6DoF parsed;
        if (!parsePacketBinary(buffer, received, parsed))
        {
            continue;
        }

        std::lock_guard<std::mutex> lock(poseMutex_);
        latestPose_ = parsed;
    }

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif
}
