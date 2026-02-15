#include "PoseFilter.h"
#include <iostream>

PoseFilter::PoseFilter()
    : initialized_(false), alpha_(0.7), windowSize_(5)
{
    initializeKalman();
}

void PoseFilter::initializeKalman()
{
    // Kalman simplificado para posição (3D) e rotação (quaternion)
    kalmanPos_.x = Eigen::VectorXd::Zero(6); // [x,y,z, vx,vy,vz]
    kalmanPos_.P = Eigen::MatrixXd::Identity(6, 6) * 1.0;
    kalmanPos_.Q = Eigen::MatrixXd::Identity(6, 6) * 0.01; // ruído do processo
    kalmanPos_.R = Eigen::MatrixXd::Identity(3, 3) * 0.1;  // ruído da medição

    kalmanRot_.x = Eigen::VectorXd::Zero(8); // [qw,qx,qy,qz, wx,wy,wz,ww]
    kalmanRot_.P = Eigen::MatrixXd::Identity(8, 8) * 1.0;
    kalmanRot_.Q = Eigen::MatrixXd::Identity(8, 8) * 0.01;
    kalmanRot_.R = Eigen::MatrixXd::Identity(4, 4) * 0.1;
}

Pose6DoF PoseFilter::filterComplementary(const Pose6DoF &measuredPose, double dt)
{
    if (!initialized_ || !filteredPose_.valid)
    {
        filteredPose_ = measuredPose;
        previousPose_ = measuredPose;
        initialized_ = true;
        return filteredPose_;
    }

    if (!measuredPose.valid)
    {
        return filteredPose_; // Manter última pose válida
    }

    // Filtro complementar: mistura medição com predição
    // pose_filtered = alpha * measurement + (1-alpha) * previous

    Pose6DoF result;
    result.timestamp = measuredPose.timestamp;
    result.valid = true;

    // Posição
    result.position = alpha_ * measuredPose.position + (1.0 - alpha_) * filteredPose_.position;

    // Rotação (SLERP para interpolar quaternions)
    result.rotation = filteredPose_.rotation.slerp(alpha_, measuredPose.rotation);
    result.rotation.normalize();

    // Atualizar matriz
    result.matrix = Eigen::Matrix4d::Identity();
    result.matrix.block<3, 3>(0, 0) = result.rotation.toRotationMatrix();
    result.matrix.block<3, 1>(0, 3) = result.position;

    filteredPose_ = result;
    previousPose_ = measuredPose;

    return result;
}

Pose6DoF PoseFilter::filterMovingAverage(const Pose6DoF &measuredPose)
{
    if (!measuredPose.valid)
    {
        return filteredPose_;
    }

    // Adicionar à fila
    poseHistory_.push_back(measuredPose);

    // Manter apenas windowSize_ elementos
    while (poseHistory_.size() > (size_t)windowSize_)
    {
        poseHistory_.pop_front();
    }

    if (poseHistory_.empty())
    {
        return measuredPose;
    }

    // Calcular média
    Pose6DoF result;
    result.timestamp = measuredPose.timestamp;
    result.valid = true;
    result.position = Eigen::Vector3d::Zero();

    // Média simples para posição
    for (const auto &pose : poseHistory_)
    {
        result.position += pose.position;
    }
    result.position /= (double)poseHistory_.size();

    // Para quaternions, usar média ponderada (simplificado)
    // Em produção, usar média de quaternions adequada
    result.rotation = poseHistory_.back().rotation;

    // Atualizar matriz
    result.matrix = Eigen::Matrix4d::Identity();
    result.matrix.block<3, 3>(0, 0) = result.rotation.toRotationMatrix();
    result.matrix.block<3, 1>(0, 3) = result.position;

    filteredPose_ = result;
    return result;
}

double PoseFilter::filterKalman1D(double measurement, double dt, KalmanState &state)
{
    // Predição
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2, 2);
    F(0, 1) = dt; // x = x + v*dt

    state.x = F * state.x;
    state.P = F * state.P * F.transpose() + state.Q.block<2, 2>(0, 0);

    // Atualização
    Eigen::VectorXd z(1);
    z(0) = measurement;

    Eigen::MatrixXd H(1, 2);
    H << 1, 0; // Medir apenas posição

    Eigen::VectorXd y = z - H * state.x;
    Eigen::MatrixXd S = H * state.P * H.transpose() + state.R.block<1, 1>(0, 0);
    Eigen::MatrixXd K = state.P * H.transpose() * S.inverse();

    state.x = state.x + K * y;
    state.P = (Eigen::MatrixXd::Identity(2, 2) - K * H) * state.P;

    return state.x(0);
}

Pose6DoF PoseFilter::filterKalman(const Pose6DoF &measuredPose, double dt)
{
    if (!initialized_ || !filteredPose_.valid)
    {
        kalmanPos_.x.head<3>() = measuredPose.position;
        kalmanPos_.x.tail<3>() = Eigen::Vector3d::Zero();

        kalmanRot_.x.head<4>() << measuredPose.rotation.w(),
            measuredPose.rotation.x(),
            measuredPose.rotation.y(),
            measuredPose.rotation.z();
        kalmanRot_.x.tail<4>() = Eigen::VectorXd::Zero(4);

        filteredPose_ = measuredPose;
        initialized_ = true;
        return filteredPose_;
    }

    if (!measuredPose.valid)
    {
        return filteredPose_;
    }

    Pose6DoF result;
    result.timestamp = measuredPose.timestamp;
    result.valid = true;

    // Filtrar cada componente de posição separadamente
    KalmanState tempState;

    // X
    tempState = kalmanPos_;
    tempState.x = kalmanPos_.x.segment<2>(0);
    result.position.x() = filterKalman1D(measuredPose.position.x(), dt, tempState);
    kalmanPos_.x.segment<2>(0) = tempState.x;

    // Y
    tempState.x = kalmanPos_.x.segment<2>(2);
    result.position.y() = filterKalman1D(measuredPose.position.y(), dt, tempState);
    kalmanPos_.x.segment<2>(2) = tempState.x;

    // Z
    tempState.x = kalmanPos_.x.segment<2>(4);
    result.position.z() = filterKalman1D(measuredPose.position.z(), dt, tempState);
    kalmanPos_.x.segment<2>(4) = tempState.x;

    // Rotação (simplificado - usar SLERP com peso)
    result.rotation = filteredPose_.rotation.slerp(0.3, measuredPose.rotation);
    result.rotation.normalize();

    // Atualizar matriz
    result.matrix = Eigen::Matrix4d::Identity();
    result.matrix.block<3, 3>(0, 0) = result.rotation.toRotationMatrix();
    result.matrix.block<3, 1>(0, 3) = result.position;

    filteredPose_ = result;
    return result;
}

void PoseFilter::reset()
{
    initialized_ = false;
    poseHistory_.clear();
    initializeKalman();
}
