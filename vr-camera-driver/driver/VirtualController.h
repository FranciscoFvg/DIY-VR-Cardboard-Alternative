#pragma once

#include <openvr_driver.h>
#include "PoseEstimator.h"
#include <string>
#include <memory>

// Controlador virtual que emula um VIVE Controller
class VirtualController : public vr::ITrackedDeviceServerDriver
{
public:
    VirtualController(int controllerId);
    virtual ~VirtualController() = default;

    // Interface ITrackedDeviceServerDriver
    virtual vr::EVRInitError Activate(uint32_t unObjectId) override;
    virtual void Deactivate() override;
    virtual void EnterStandby() override;
    virtual void *GetComponent(const char *pchComponentNameAndVersion) override;
    virtual void DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize) override;
    virtual vr::DriverPose_t GetPose() override;

    // Atualizar pose do controlador
    void UpdatePose(const Pose6DoF &pose);

    // Simular input de botões
    void UpdateButtonState(vr::EVRButtonId button, bool pressed);
    void UpdateTrigger(float value);
    void UpdateTrackpad(float x, float y, bool touched);

    uint32_t GetDeviceIndex() const { return objectId_; }
    bool IsActivated() const { return objectId_ != vr::k_unTrackedDeviceIndexInvalid; }
    const std::string &GetSerialNumber() const { return serialNumber_; }

private:
    uint32_t objectId_;
    int controllerId_;
    vr::DriverPose_t currentPose_;
    vr::PropertyContainerHandle_t propertyContainer_;
    vr::VRInputComponentHandle_t triggerClickHandle_;
    vr::VRInputComponentHandle_t triggerValueHandle_;
    float lastTriggerValue_;
    bool lastTriggerPressed_;

    std::string serialNumber_;
    std::string modelNumber_;
};
