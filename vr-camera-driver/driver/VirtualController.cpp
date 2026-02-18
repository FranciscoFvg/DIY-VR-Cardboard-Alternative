#include "VirtualController.h"
#include <cmath>
#include <cstring>
#include <iostream>

using namespace vr;

VirtualController::VirtualController(int controllerId)
    : objectId_(k_unTrackedDeviceIndexInvalid), controllerId_(controllerId), propertyContainer_(k_ulInvalidPropertyContainer),
    triggerClickHandle_(0), triggerValueHandle_(0), lastTriggerValue_(0.0f), lastTriggerPressed_(false)
{

    serialNumber_ = "CAMERA_CONTROLLER_" + std::to_string(controllerId_);
    modelNumber_ = "Vive Controller";

    // Inicializar pose padrão
    currentPose_ = {};
    currentPose_.poseIsValid = false;
    currentPose_.result = TrackingResult_Uninitialized;
    currentPose_.deviceIsConnected = true;

    // Matriz identidade
    currentPose_.qWorldFromDriverRotation.w = 1;
    currentPose_.qWorldFromDriverRotation.x = 0;
    currentPose_.qWorldFromDriverRotation.y = 0;
    currentPose_.qWorldFromDriverRotation.z = 0;

    currentPose_.vecWorldFromDriverTranslation[0] = 0;
    currentPose_.vecWorldFromDriverTranslation[1] = 0;
    currentPose_.vecWorldFromDriverTranslation[2] = 0;

    currentPose_.qDriverFromHeadRotation.w = 1;
    currentPose_.qDriverFromHeadRotation.x = 0;
    currentPose_.qDriverFromHeadRotation.y = 0;
    currentPose_.qDriverFromHeadRotation.z = 0;

    currentPose_.vecDriverFromHeadTranslation[0] = 0;
    currentPose_.vecDriverFromHeadTranslation[1] = 0;
    currentPose_.vecDriverFromHeadTranslation[2] = 0;
}

EVRInitError VirtualController::Activate(uint32_t unObjectId)
{
    objectId_ = unObjectId;
    propertyContainer_ = VRProperties()->TrackedDeviceToPropertyContainer(objectId_);

    // Configurar propriedades do dispositivo
    VRProperties()->SetStringProperty(propertyContainer_, Prop_SerialNumber_String, serialNumber_.c_str());
    VRProperties()->SetStringProperty(propertyContainer_, Prop_ModelNumber_String, modelNumber_.c_str());
    VRProperties()->SetStringProperty(propertyContainer_, Prop_ControllerType_String, "vive_controller");
    VRProperties()->SetStringProperty(propertyContainer_, Prop_ManufacturerName_String, "CameraVR");
    VRProperties()->SetStringProperty(propertyContainer_, Prop_RenderModelName_String, "vr_controller_vive_1_5");

    VRProperties()->SetInt32Property(propertyContainer_, Prop_DeviceClass_Int32, TrackedDeviceClass_Controller);
    VRProperties()->SetInt32Property(propertyContainer_, Prop_ControllerRoleHint_Int32,
                                     controllerId_ == 0 ? TrackedControllerRole_LeftHand : TrackedControllerRole_RightHand);

    // Input profile
    VRProperties()->SetStringProperty(propertyContainer_, Prop_InputProfilePath_String,
                                      "{htc}/input/vive_controller_profile.json");

    // Capacidades
    VRProperties()->SetBoolProperty(propertyContainer_, Prop_WillDriftInYaw_Bool, false);
    VRProperties()->SetBoolProperty(propertyContainer_, Prop_DeviceProvidesBatteryStatus_Bool, false);
    VRProperties()->SetBoolProperty(propertyContainer_, Prop_DeviceCanPowerOff_Bool, false);

    VRDriverInput()->CreateBooleanComponent(propertyContainer_, "/input/trigger/click", &triggerClickHandle_);
    VRDriverInput()->CreateScalarComponent(propertyContainer_, "/input/trigger/value", &triggerValueHandle_,
                                           VRScalarType_Absolute, VRScalarUnits_NormalizedOneSided);

    return VRInitError_None;
}

void VirtualController::Deactivate()
{
    objectId_ = k_unTrackedDeviceIndexInvalid;
}

void VirtualController::EnterStandby()
{
}

void *VirtualController::GetComponent(const char *pchComponentNameAndVersion)
{
    return nullptr;
}

void VirtualController::DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize)
{
    if (unResponseBufferSize > 0)
    {
        pchResponseBuffer[0] = 0;
    }
}

DriverPose_t VirtualController::GetPose()
{
    return currentPose_;
}

void VirtualController::UpdatePose(const Pose6DoF &pose)
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

    // Posição
    currentPose_.vecPosition[0] = pose.position.x();
    currentPose_.vecPosition[1] = pose.position.y();
    currentPose_.vecPosition[2] = pose.position.z();

    // Rotação (quaternion)
    currentPose_.qRotation.w = pose.rotation.w();
    currentPose_.qRotation.x = pose.rotation.x();
    currentPose_.qRotation.y = pose.rotation.y();
    currentPose_.qRotation.z = pose.rotation.z();

    // Velocidades (zero por enquanto)
    currentPose_.vecVelocity[0] = 0;
    currentPose_.vecVelocity[1] = 0;
    currentPose_.vecVelocity[2] = 0;

    currentPose_.vecAngularVelocity[0] = 0;
    currentPose_.vecAngularVelocity[1] = 0;
    currentPose_.vecAngularVelocity[2] = 0;

    // Notificar SteamVR da mudança de pose
    if (objectId_ != k_unTrackedDeviceIndexInvalid)
    {
        if (VRServerDriverHost())
        {
            VRServerDriverHost()->TrackedDevicePoseUpdated(objectId_, currentPose_, sizeof(DriverPose_t));
            std::cout << "[VirtualController " << controllerId_ << "] Pose atualizada: pos=(" 
                      << pose.position.x() << ", " << pose.position.y() << ", " << pose.position.z() << ")" << std::endl;
        }
        else
        {
            std::cerr << "[VirtualController " << controllerId_ << "] ERRO: VRServerDriverHost() é NULL!" << std::endl;
        }
    }
}

void VirtualController::UpdateButtonState(EVRButtonId button, bool pressed)
{
    if (objectId_ == k_unTrackedDeviceIndexInvalid)
        return;

    if (button == k_EButton_SteamVR_Trigger && triggerClickHandle_ != 0)
    {
        VRDriverInput()->UpdateBooleanComponent(triggerClickHandle_, pressed, 0);
    }
}

void VirtualController::UpdateTrigger(float value)
{
    if (objectId_ == k_unTrackedDeviceIndexInvalid)
        return;

    float clamped = value;
    if (clamped < 0.0f)
        clamped = 0.0f;
    if (clamped > 1.0f)
        clamped = 1.0f;

    if (triggerValueHandle_ != 0)
    {
        VRDriverInput()->UpdateScalarComponent(triggerValueHandle_, clamped, 0);
    }

    bool pressed = clamped >= 0.5f;
    if (triggerClickHandle_ != 0)
    {
        VRDriverInput()->UpdateBooleanComponent(triggerClickHandle_, pressed, 0);
    }

    if (pressed != lastTriggerPressed_)
    {
        std::cout << "[VirtualController " << controllerId_ << "] Trigger "
                  << (pressed ? "PRESSED" : "RELEASED")
                  << " (value=" << clamped << ")" << std::endl;
        lastTriggerPressed_ = pressed;
    }

    if (std::fabs(clamped - lastTriggerValue_) >= 0.02f)
    {
        std::cout << "[VirtualController " << controllerId_ << "] Trigger value="
                  << clamped << std::endl;
        lastTriggerValue_ = clamped;
    }
}

void VirtualController::UpdateTrackpad(float x, float y, bool touched)
{
    if (objectId_ == k_unTrackedDeviceIndexInvalid)
        return;

    // Implementar quando adicionar input components
}
