#pragma once

#include <Eigen/Dense>

struct Pose6DoF
{
    Eigen::Vector3d position;    // x, y, z em metros
    Eigen::Quaterniond rotation; // quaternion
    Eigen::Matrix4d matrix;      // matriz 4x4 de transformação
    double timestamp;
    bool valid;

    Pose6DoF() : position(0, 0, 0), rotation(1, 0, 0, 0),
                 matrix(Eigen::Matrix4d::Identity()),
                 timestamp(0), valid(false) {}
};

class PoseEstimator
{
public:
    PoseEstimator();

    // Converter pose a partir de posicao + quaternion (MediaPipe)
    Pose6DoF convertFromHandPose(const Eigen::Vector3d &position, const Eigen::Quaterniond &rotation, double timestamp);

    // Converter Pose6DoF para matriz OpenVR (3x4)
    void toOpenVRMatrix(const Pose6DoF &pose, float *matrix);

    // Transformar pose de coordenadas da câmera para coordenadas do mundo VR
    Pose6DoF transformToVRSpace(const Pose6DoF &cameraPose);

    // Inverter pose (útil para converter pose do marcador em pose da câmera)
    Pose6DoF invertPose(const Pose6DoF &pose);

    void setCalibrationOffset(const Eigen::Vector3d &offset, const Eigen::Quaterniond &rotOffset);

private:
    Eigen::Vector3d calibrationOffset_;
    Eigen::Quaterniond calibrationRotation_;
};
