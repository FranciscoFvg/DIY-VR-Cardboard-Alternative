#pragma once

#include <openvr_driver.h>
#include "PoseEstimator.h"
#include <string>

class VirtualHMD : public vr::ITrackedDeviceServerDriver, public vr::IVRDisplayComponent
{
public:
    VirtualHMD();
    virtual ~VirtualHMD() = default;

    virtual vr::EVRInitError Activate(uint32_t unObjectId) override;
    virtual void Deactivate() override;
    virtual void EnterStandby() override;
    virtual void *GetComponent(const char *pchComponentNameAndVersion) override;
    virtual void DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize) override;
    virtual vr::DriverPose_t GetPose() override;

    // Interface IVRDisplayComponent
    virtual void GetWindowBounds(int32_t *pnX, int32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight) override;
    virtual bool IsDisplayOnDesktop() override;
    virtual bool IsDisplayRealDisplay() override;
    virtual void GetRecommendedRenderTargetSize(uint32_t *pnWidth, uint32_t *pnHeight) override;
    virtual void GetEyeOutputViewport(vr::EVREye eEye, uint32_t *pnX, uint32_t *pnY,
                                      uint32_t *pnWidth, uint32_t *pnHeight) override;
    virtual void GetProjectionRaw(vr::EVREye eEye, float *pfLeft, float *pfRight,
                                  float *pfTop, float *pfBottom) override;
    virtual vr::DistortionCoordinates_t ComputeDistortion(vr::EVREye eEye, float fU, float fV) override;
    virtual bool ComputeInverseDistortion(vr::HmdVector2_t *pResult, vr::EVREye eEye,
                                          uint32_t unChannel, float fU, float fV) override;

    void UpdatePose(const Pose6DoF &pose);

    uint32_t GetDeviceIndex() const { return objectId_; }
    bool IsActivated() const { return objectId_ != vr::k_unTrackedDeviceIndexInvalid; }
    const std::string &GetSerialNumber() const { return serialNumber_; }

private:
    uint32_t objectId_;
    vr::DriverPose_t currentPose_;
    vr::PropertyContainerHandle_t propertyContainer_;
    std::string serialNumber_;
    std::string modelNumber_;
};
