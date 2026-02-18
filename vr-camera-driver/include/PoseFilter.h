#pragma once

#include "PoseEstimator.h"
#include <Eigen/Dense>
#include <deque>

class PoseFilter
{
public:
    PoseFilter();

    // Filtro complementar simples (mistura giroscópio virtual com visão)
    Pose6DoF filterComplementary(const Pose6DoF &measuredPose, double dt);

    // Filtro de média móvel para suavizar jitter
    Pose6DoF filterMovingAverage(const Pose6DoF &measuredPose);

    // Filtro Kalman simplificado (1D para cada componente)
    Pose6DoF filterKalman(const Pose6DoF &measuredPose, double dt);

    void reset();
    void setAlpha(double alpha) { alpha_ = alpha; }      // Para complementar (0-1)
    void setWindowSize(int size) { windowSize_ = size; } // Para média móvel
    void setPositionDeadband(double meters) { positionDeadband_ = meters; }
    void setRotationDeadbandRad(double radians) { rotationDeadbandRad_ = radians; }
    void setMaxPositionSpeed(double metersPerSecond) { maxPositionSpeed_ = metersPerSecond; }
    void setMaxRotationSpeedRad(double radiansPerSecond) { maxRotationSpeedRad_ = radiansPerSecond; }

private:
    // Estado do filtro
    Pose6DoF previousPose_;
    Pose6DoF filteredPose_;
    bool initialized_;

    // Parâmetros
    double alpha_; // Complementar: quanto confiar na medição (vs integração)
    int windowSize_;
    double positionDeadband_;
    double rotationDeadbandRad_;
    double maxPositionSpeed_;
    double maxRotationSpeedRad_;

    // Histórico para média móvel
    std::deque<Pose6DoF> poseHistory_;

    // Estado Kalman (simplificado)
    struct KalmanState
    {
        Eigen::VectorXd x; // estado [pos, vel]
        Eigen::MatrixXd P; // covariância
        Eigen::MatrixXd Q; // ruído do processo
        Eigen::MatrixXd R; // ruído da medição
    };

    KalmanState kalmanPos_;
    KalmanState kalmanRot_;

    void initializeKalman();
    double filterKalman1D(double measurement, double dt, KalmanState &state);
};
