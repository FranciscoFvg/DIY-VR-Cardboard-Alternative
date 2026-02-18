#include "CameraVRDriver.h"
#include "../include/HandPoseReceiver.h"
#include "../include/ArduinoHeadReceiver.h"
#include "../include/PoseEstimator.h"
#include "../include/PoseFilter.h"
#include <iostream>
#include <cstring>
#include <chrono>

using namespace vr;

CameraVRDriver g_CameraVRDriver;

CameraVRDriver::CameraVRDriver()
    : trackingRunning_(false)
{
}

CameraVRDriver::~CameraVRDriver()
{
    StopTrackingThread();
}

EVRInitError CameraVRDriver::Init(IVRDriverContext *pDriverContext)
{
    VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);

    std::cout << "========================================" << std::endl;
    std::cout << "CameraVR Driver inicializado com sucesso!" << std::endl;
    std::cout << "VRServerDriverHost: " << (VRServerDriverHost() ? "VÁLIDO" : "NULL") << std::endl;
    std::cout << "========================================" << std::endl;

    AddHmd();

    // Adicionar 2 controladores por padrão
    AddController(0); // Esquerdo
    AddController(1); // Direito

    // Iniciar thread de tracking
    StartTrackingThread();

    return VRInitError_None;
}

void CameraVRDriver::Cleanup()
{
    StopTrackingThread();
    hmd_.reset();
    controllers_.clear();
    VR_CLEANUP_SERVER_DRIVER_CONTEXT();
}

const char *const *CameraVRDriver::GetInterfaceVersions()
{
    return k_InterfaceVersions;
}

void CameraVRDriver::RunFrame()
{
    // Atualizar lógica dos controladores aqui
    // (A pose é atualizada externamente via UpdatePose)
}

bool CameraVRDriver::ShouldBlockStandbyMode()
{
    return false;
}

void CameraVRDriver::EnterStandby()
{
}

void CameraVRDriver::LeaveStandby()
{
}

void CameraVRDriver::AddHmd()
{
    hmd_ = std::make_shared<VirtualHMD>();

    if (!VRServerDriverHost())
    {
        std::cerr << "ERRO: VRServerDriverHost() é NULL ao tentar adicionar HMD" << std::endl;
        hmd_.reset();
        return;
    }

    bool success = VRServerDriverHost()->TrackedDeviceAdded(
        hmd_->GetSerialNumber().c_str(),
        TrackedDeviceClass_HMD,
        hmd_.get());

    if (success)
    {
        std::cout << "[CameraVRDriver] HMD virtual adicionado com sucesso!" << std::endl;
    }
    else
    {
        std::cerr << "[CameraVRDriver] ERRO: Falha ao adicionar HMD virtual" << std::endl;
        hmd_.reset();
    }
}

void CameraVRDriver::AddController(int controllerId)
{
    auto controller = std::make_shared<VirtualController>(controllerId);

    // Adicionar ao SteamVR
    if (!VRServerDriverHost())
    {
        std::cerr << "ERRO: VRServerDriverHost() é NULL ao tentar adicionar controlador " << controllerId << std::endl;
        return;
    }

    bool success = VRServerDriverHost()->TrackedDeviceAdded(
        controller->GetSerialNumber().c_str(),
        TrackedDeviceClass_Controller,
        controller.get());

    if (success)
    {
        controllers_.push_back(controller);
        std::cout << "[CameraVRDriver] Controlador " << controllerId << " adicionado com sucesso!" << std::endl;
    }
    else
    {
        std::cerr << "[CameraVRDriver] ERRO: Falha ao adicionar controlador " << controllerId << std::endl;
    }
}

VirtualController *CameraVRDriver::GetController(int controllerId)
{
    if (controllerId >= 0 && controllerId < (int)controllers_.size())
    {
        return controllers_[controllerId].get();
    }
    return nullptr;
}

// Entry points do driver OpenVR
extern "C" __declspec(dllexport) void *HmdDriverFactory(const char *pInterfaceName, int *pReturnCode)
{
    if (0 == strcmp(IServerTrackedDeviceProvider_Version, pInterfaceName))
    {
        return &g_CameraVRDriver;
    }

    if (pReturnCode)
    {
        *pReturnCode = VRInitError_Init_InterfaceNotFound;
    }

    return nullptr;
}

void CameraVRDriver::StartTrackingThread()
{
    if (trackingRunning_)
        return;

    std::cout << "[CameraVRDriver] Iniciando thread de tracking..." << std::endl;

    receiver_ = std::make_unique<HandPoseReceiver>(7000);
    receiver_->start();

    arduinoHeadReceiver_ = std::make_unique<ArduinoHeadReceiver>(4242);
    arduinoHeadReceiver_->start();

    poseEstimator_ = std::make_unique<PoseEstimator>();
    leftFilter_ = std::make_unique<PoseFilter>();
    rightFilter_ = std::make_unique<PoseFilter>();
    leftFilter_->setAlpha(0.7);
    rightFilter_->setAlpha(0.7);

    trackingRunning_ = true;
    trackingThread_ = std::thread(&CameraVRDriver::TrackingLoop, this);

    std::cout << "[CameraVRDriver] Thread de tracking iniciada!" << std::endl;
}

void CameraVRDriver::StopTrackingThread()
{
    if (!trackingRunning_)
        return;

    std::cout << "[CameraVRDriver] Parando thread de tracking..." << std::endl;

    trackingRunning_ = false;
    if (trackingThread_.joinable())
        trackingThread_.join();

    if (receiver_)
        receiver_->stop();

    if (arduinoHeadReceiver_)
        arduinoHeadReceiver_->stop();

    std::cout << "[CameraVRDriver] Thread de tracking parada!" << std::endl;
}

void CameraVRDriver::TrackingLoop()
{
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (trackingRunning_)
    {
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;

        if (!receiver_)
            break;

        if (hmd_ && arduinoHeadReceiver_)
        {
            Pose6DoF headPose = arduinoHeadReceiver_->getLatestPose();
            hmd_->UpdatePose(headPose);
        }

        Pose6DoF leftPose = receiver_->getLatestPose(HandPoseReceiver::Hand::Left);
        Pose6DoF rightPose = receiver_->getLatestPose(HandPoseReceiver::Hand::Right);
        float leftTrigger = receiver_->getLatestTrigger(HandPoseReceiver::Hand::Left);
        float rightTrigger = receiver_->getLatestTrigger(HandPoseReceiver::Hand::Right);

        if (VirtualController *leftController = GetController(0))
        {
            Pose6DoF pose = leftPose.valid ? poseEstimator_->transformToVRSpace(leftPose) : leftPose;
            pose = leftPose.valid ? leftFilter_->filterComplementary(pose, dt) : pose;
            leftController->UpdatePose(pose);
            leftController->UpdateTrigger(leftPose.valid ? leftTrigger : 0.0f);
        }

        if (VirtualController *rightController = GetController(1))
        {
            Pose6DoF pose = rightPose.valid ? poseEstimator_->transformToVRSpace(rightPose) : rightPose;
            pose = rightPose.valid ? rightFilter_->filterComplementary(pose, dt) : pose;
            rightController->UpdatePose(pose);
            rightController->UpdateTrigger(rightPose.valid ? rightTrigger : 0.0f);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

