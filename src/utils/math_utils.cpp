#include "utils/math_utils.h"
#include <cmath>

namespace endorobo {
namespace math_utils {

Eigen::Vector3d rotationMatrixToEulerAngles(const Eigen::Matrix3d& R) {
    Eigen::Vector3d euler;
    
    // Roll (x-axis rotation)
    euler(0) = std::atan2(R(2, 1), R(2, 2));
    
    // Pitch (y-axis rotation)
    euler(1) = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    
    // Yaw (z-axis rotation)
    euler(2) = std::atan2(R(1, 0), R(0, 0));
    
    return euler;
}

Eigen::Matrix3d eulerAnglesToRotationMatrix(const Eigen::Vector3d& euler) {
    double roll = euler(0);
    double pitch = euler(1);
    double yaw = euler(2);
    
    // Roll (X)
    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0,
          0, std::cos(roll), -std::sin(roll),
          0, std::sin(roll), std::cos(roll);
    
    // Pitch (Y)
    Eigen::Matrix3d Ry;
    Ry << std::cos(pitch), 0, std::sin(pitch),
          0, 1, 0,
          -std::sin(pitch), 0, std::cos(pitch);
    
    // Yaw (Z)
    Eigen::Matrix3d Rz;
    Rz << std::cos(yaw), -std::sin(yaw), 0,
          std::sin(yaw), std::cos(yaw), 0,
          0, 0, 1;
    
    return Rz * Ry * Rx;
}

Eigen::Matrix3d quaternionToRotationMatrix(const Eigen::Quaterniond& q) {
    return q.normalized().toRotationMatrix();
}

Eigen::Quaterniond rotationMatrixToQuaternion(const Eigen::Matrix3d& R) {
    return Eigen::Quaterniond(R);
}

Eigen::Matrix4d relativeTransform(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2) {
    return T1.inverse() * T2;
}

Eigen::Matrix4d interpolateTransform(const Eigen::Matrix4d& T1,
                                     const Eigen::Matrix4d& T2,
                                     double t) {
    // 插值平移部分
    Eigen::Vector3d t1 = T1.block<3, 1>(0, 3);
    Eigen::Vector3d t2 = T2.block<3, 1>(0, 3);
    Eigen::Vector3d t_interp = lerp(t1, t2, t);
    
    // 插值旋转部分（使用四元数球面线性插值）
    Eigen::Quaterniond q1(T1.block<3, 3>(0, 0));
    Eigen::Quaterniond q2(T2.block<3, 3>(0, 0));
    Eigen::Quaterniond q_interp = q1.slerp(t, q2);
    
    // 构建插值后的变换矩阵
    Eigen::Matrix4d T_interp = Eigen::Matrix4d::Identity();
    T_interp.block<3, 3>(0, 0) = q_interp.toRotationMatrix();
    T_interp.block<3, 1>(0, 3) = t_interp;
    
    return T_interp;
}

} // namespace math_utils
} // namespace endorobo

