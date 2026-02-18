#include "VirtualHMD.h"
#include <iostream>
#include <cstring>

using namespace vr;

VirtualHMD::VirtualHMD()
    : objectId_(k_unTrackedDeviceIndexInvalid), propertyContainer_(k_ulInvalidPropertyContainer)
{
    serialNumber_ = "CAMERA_HMD_0";
    modelNumber_ = "CameraVR HMD";

    currentPose_ = {};
    currentPose_.poseIsValid = false;
    currentPose_.result = TrackingResult_Uninitialized;
    currentPose_.deviceIsConnected = true;

    currentPose_.qWorldFromDriverRotation.w = 1;
    currentPose_.qDriverFromHeadRotation.w = 1;
}

EVRInitError VirtualHMD::Activate(uint32_t unObjectId)
{
    objectId_ = unObjectId;
    propertyContainer_ = VRProperties()->TrackedDeviceToPropertyContainer(objectId_);

    VRProperties()->SetStringProperty(propertyContainer_, Prop_SerialNumber_String, serialNumber_.c_str());
    VRProperties()->SetStringProperty(propertyContainer_, Prop_ModelNumber_String, modelNumber_.c_str());
    VRProperties()->SetStringProperty(propertyContainer_, Prop_ManufacturerName_String, "CameraVR");
    VRProperties()->SetStringProperty(propertyContainer_, Prop_TrackingSystemName_String, "cameravr");
    VRProperties()->SetBoolProperty(propertyContainer_, Prop_IsOnDesktop_Bool, false);
    VRProperties()->SetBoolProperty(propertyContainer_, Prop_DeviceProvidesBatteryStatus_Bool, false);
    VRProperties()->SetBoolProperty(propertyContainer_, Prop_DeviceCanPowerOff_Bool, false);
    VRProperties()->SetFloatProperty(propertyContainer_, Prop_DisplayFrequency_Float, 60.0f);
    VRProperties()->SetFloatProperty(propertyContainer_, Prop_SecondsFromVsyncToPhotons_Float, 0.011f);
    VRProperties()->SetFloatProperty(propertyContainer_, Prop_UserIpdMeters_Float, 0.063f);
    VRProperties()->SetFloatProperty(propertyContainer_, Prop_UserHeadToEyeDepthMeters_Float, 0.0f);
    VRProperties()->SetStringProperty(propertyContainer_, Prop_RenderModelName_String, "generic_hmd");
    VRProperties()->SetBoolProperty(propertyContainer_, Prop_HasDisplayComponent_Bool, true);
    VRProperties()->SetUint64Property(propertyContainer_, Prop_CurrentUniverseId_Uint64, 2);

    return VRInitError_None;
}

void VirtualHMD::Deactivate()
{
    objectId_ = k_unTrackedDeviceIndexInvalid;
}

void VirtualHMD::EnterStandby()
{
}

void *VirtualHMD::GetComponent(const char *pchComponentNameAndVersion)
{
    if (0 == std::strcmp(pchComponentNameAndVersion, IVRDisplayComponent_Version))
    {
        return static_cast<IVRDisplayComponent *>(this);
    }

    return nullptr;
}

void VirtualHMD::DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize)
{
    if (unResponseBufferSize > 0)
    {
        pchResponseBuffer[0] = 0;
    }
}

DriverPose_t VirtualHMD::GetPose()
{
    return currentPose_;
}

void VirtualHMD::GetWindowBounds(int32_t *pnX, int32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight)
{
    *pnX = 0;
    *pnY = 0;
    *pnWidth = 1920;
    *pnHeight = 1080;
}

bool VirtualHMD::IsDisplayOnDesktop()
{
    return true;
}

bool VirtualHMD::IsDisplayRealDisplay()
{
    return false;
}

void VirtualHMD::GetRecommendedRenderTargetSize(uint32_t *pnWidth, uint32_t *pnHeight)
{
    *pnWidth = 1512;
    *pnHeight = 1680;
}

void VirtualHMD::GetEyeOutputViewport(EVREye eEye, uint32_t *pnX, uint32_t *pnY,
                                      uint32_t *pnWidth, uint32_t *pnHeight)
{
    *pnY = 0;
    *pnWidth = 960;
    *pnHeight = 1080;
    *pnX = (eEye == Eye_Left) ? 0 : 960;
}

void VirtualHMD::GetProjectionRaw(EVREye, float *pfLeft, float *pfRight,
                                  float *pfTop, float *pfBottom)
{
    *pfLeft = -1.0f;
    *pfRight = 1.0f;
    *pfTop = -1.0f;
    *pfBottom = 1.0f;
}

DistortionCoordinates_t VirtualHMD::ComputeDistortion(EVREye, float fU, float fV)
{
    DistortionCoordinates_t coords = {};
    coords.rfRed[0] = fU;
    coords.rfRed[1] = fV;
    coords.rfGreen[0] = fU;
    coords.rfGreen[1] = fV;
    coords.rfBlue[0] = fU;
    coords.rfBlue[1] = fV;
    return coords;
}

bool VirtualHMD::ComputeInverseDistortion(HmdVector2_t *pResult, EVREye,
                                          uint32_t, float fU, float fV)
{
    if (!pResult)
    {
        return false;
    }

    pResult->v[0] = fU;
    pResult->v[1] = fV;
    return true;
}

void VirtualHMD::UpdatePose(const Pose6DoF &pose)
{
    if (!pose.valid)
    {
        currentPose_.poseIsValid = false;
        currentPose_.result = TrackingResult_Running_OutOfRange;
        return;
    }

    currentPose_.poseIsValid = true;
    currentPose_.result = TrackingResult_Running_OK;
    currentPose_.deviceIsConnected = true;

    currentPose_.vecPosition[0] = pose.position.x();
    currentPose_.vecPosition[1] = pose.position.y();
    currentPose_.vecPosition[2] = pose.position.z();

    currentPose_.qRotation.w = pose.rotation.w();
    currentPose_.qRotation.x = pose.rotation.x();
    currentPose_.qRotation.y = pose.rotation.y();
    currentPose_.qRotation.z = pose.rotation.z();

    currentPose_.vecVelocity[0] = 0;
    currentPose_.vecVelocity[1] = 0;
    currentPose_.vecVelocity[2] = 0;
    currentPose_.vecAngularVelocity[0] = 0;
    currentPose_.vecAngularVelocity[1] = 0;
    currentPose_.vecAngularVelocity[2] = 0;

    if (objectId_ != k_unTrackedDeviceIndexInvalid && VRServerDriverHost())
    {
        VRServerDriverHost()->TrackedDevicePoseUpdated(objectId_, currentPose_, sizeof(DriverPose_t));
    }
}
