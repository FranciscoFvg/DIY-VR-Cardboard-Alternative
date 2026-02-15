#pragma once

#include <openvr_driver.h>
#include "VirtualController.h"
#include <vector>
#include <memory>
#include <thread>
#include <atomic>

class HandPoseReceiver;
class PoseEstimator;
class PoseFilter;

class CameraVRDriver : public vr::IServerTrackedDeviceProvider
{
public:
    CameraVRDriver();
    virtual ~CameraVRDriver();

    // Interface IServerTrackedDeviceProvider
    virtual vr::EVRInitError Init(vr::IVRDriverContext *pDriverContext) override;
    virtual void Cleanup() override;
    virtual const char *const *GetInterfaceVersions() override;
    virtual void RunFrame() override;
    virtual bool ShouldBlockStandbyMode() override;
    virtual void EnterStandby() override;
    virtual void LeaveStandby() override;

    // Adicionar/remover controladores
    void AddController(int controllerId);
    VirtualController *GetController(int controllerId);

private:
    void StartTrackingThread();
    void StopTrackingThread();
    void TrackingLoop();

    std::vector<std::shared_ptr<VirtualController>> controllers_;
    std::unique_ptr<HandPoseReceiver> receiver_;
    std::unique_ptr<PoseEstimator> poseEstimator_;
    std::unique_ptr<PoseFilter> leftFilter_;
    std::unique_ptr<PoseFilter> rightFilter_;
    std::thread trackingThread_;
    std::atomic<bool> trackingRunning_;
};

// Singleton global do driver
extern CameraVRDriver g_CameraVRDriver;
