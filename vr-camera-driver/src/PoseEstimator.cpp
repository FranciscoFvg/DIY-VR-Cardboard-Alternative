#include "PoseEstimator.h"
#include <iostream>

PoseEstimator::PoseEstimator()
    : calibrationOffset_(0, 0, 0),
      calibrationRotation_(1, 0, 0, 0)
{
}

Pose6DoF PoseEstimator::convertFromHandPose(const Eigen::Vector3d &position,
                                            const Eigen::Quaterniond &rotation,
                                            double timestamp)
{
    Pose6DoF pose;
    pose.timestamp = timestamp;
    pose.valid = true;
    pose.position = position;
    pose.rotation = rotation.normalized();

    pose.matrix = Eigen::Matrix4d::Identity();
    pose.matrix.block<3, 3>(0, 0) = pose.rotation.toRotationMatrix();
    pose.matrix.block<3, 1>(0, 3) = pose.position;

    return pose;
}

void PoseEstimator::toOpenVRMatrix(const Pose6DoF &pose, float *matrix)
{
    // OpenVR usa matriz 3x4 (row-major)
    // [R | t] onde R é 3x3 rotação e t é 3x1 translação

    Eigen::Matrix3d rot = pose.rotation.toRotationMatrix();

    // Preencher matriz no formato OpenVR (row-major)
    matrix[0] = (float)rot(0, 0);
    matrix[1] = (float)rot(0, 1);
    matrix[2] = (float)rot(0, 2);
    matrix[3] = (float)pose.position.x();
    matrix[4] = (float)rot(1, 0);
    matrix[5] = (float)rot(1, 1);
    matrix[6] = (float)rot(1, 2);
    matrix[7] = (float)pose.position.y();
    matrix[8] = (float)rot(2, 0);
    matrix[9] = (float)rot(2, 1);
    matrix[10] = (float)rot(2, 2);
    matrix[11] = (float)pose.position.z();
}

Pose6DoF PoseEstimator::transformToVRSpace(const Pose6DoF &cameraPose)
{
    Pose6DoF vrPose = cameraPose;

    // Aplicar transformação de coordenadas
    // OpenCV: X direita, Y baixo, Z frente
    // SteamVR: X direita, Y cima, Z trás

    // Inverter Y e Z
    vrPose.position.y() = -cameraPose.position.y();
    vrPose.position.z() = -cameraPose.position.z();

    // Ajustar rotação
    Eigen::Matrix3d rot = cameraPose.rotation.toRotationMatrix();
    rot(1, 0) = -rot(1, 0);
    rot(1, 1) = -rot(1, 1);
    rot(1, 2) = -rot(1, 2);
    rot(2, 0) = -rot(2, 0);
    rot(2, 1) = -rot(2, 1);
    rot(2, 2) = -rot(2, 2);

    vrPose.rotation = Eigen::Quaterniond(rot);
    vrPose.rotation.normalize();

    // Aplicar offset de calibração
    vrPose.position = calibrationRotation_ * vrPose.position + calibrationOffset_;
    vrPose.rotation = calibrationRotation_ * vrPose.rotation;

    // Atualizar matriz
    vrPose.matrix = Eigen::Matrix4d::Identity();
    vrPose.matrix.block<3, 3>(0, 0) = vrPose.rotation.toRotationMatrix();
    vrPose.matrix.block<3, 1>(0, 3) = vrPose.position;

    return vrPose;
}

Pose6DoF PoseEstimator::invertPose(const Pose6DoF &pose)
{
    Pose6DoF inverted;
    inverted.timestamp = pose.timestamp;
    inverted.valid = pose.valid;

    // Inverter rotação
    inverted.rotation = pose.rotation.inverse();

    // Inverter posição: p' = -R^T * p
    inverted.position = -(inverted.rotation * pose.position);

    // Atualizar matriz
    inverted.matrix = Eigen::Matrix4d::Identity();
    inverted.matrix.block<3, 3>(0, 0) = inverted.rotation.toRotationMatrix();
    inverted.matrix.block<3, 1>(0, 3) = inverted.position;

    return inverted;
}

void PoseEstimator::setCalibrationOffset(const Eigen::Vector3d &offset,
                                         const Eigen::Quaterniond &rotOffset)
{
    calibrationOffset_ = offset;
    calibrationRotation_ = rotOffset;
    calibrationRotation_.normalize();

    std::cout << "Offset de calibração configurado: "
              << offset.transpose() << std::endl;
}
