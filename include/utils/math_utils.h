#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace endorobo {
namespace math_utils {

/**
 * @brief 将旋转矩阵转换为欧拉角(Roll, Pitch, Yaw)
 * @param R 3x3旋转矩阵
 * @return 欧拉角向量 [roll, pitch, yaw]（弧度）
 */
Eigen::Vector3d rotationMatrixToEulerAngles(const Eigen::Matrix3d& R);

/**
 * @brief 将欧拉角转换为旋转矩阵
 * @param euler 欧拉角向量 [roll, pitch, yaw]（弧度）
 * @return 3x3旋转矩阵
 */
Eigen::Matrix3d eulerAnglesToRotationMatrix(const Eigen::Vector3d& euler);

/**
 * @brief 将四元数转换为旋转矩阵
 * @param q 四元数
 * @return 3x3旋转矩阵
 */
Eigen::Matrix3d quaternionToRotationMatrix(const Eigen::Quaterniond& q);

/**
 * @brief 将旋转矩阵转换为四元数
 * @param R 3x3旋转矩阵
 * @return 四元数
 */
Eigen::Quaterniond rotationMatrixToQuaternion(const Eigen::Matrix3d& R);

/**
 * @brief 计算两个变换矩阵之间的相对变换
 * @param T1 变换矩阵1
 * @param T2 变换矩阵2
 * @return 相对变换矩阵
 */
Eigen::Matrix4d relativeTransform(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2);

/**
 * @brief 插值两个变换矩阵
 * @param T1 变换矩阵1
 * @param T2 变换矩阵2
 * @param t 插值参数 [0, 1]
 * @return 插值后的变换矩阵
 */
Eigen::Matrix4d interpolateTransform(const Eigen::Matrix4d& T1,
                                     const Eigen::Matrix4d& T2,
                                     double t);

/**
 * @brief 计算两点之间的欧几里得距离
 */
template<typename T>
inline double euclideanDistance(const T& p1, const T& p2) {
    return (p1 - p2).norm();
}

/**
 * @brief 角度转弧度
 */
inline double degToRad(double degrees) {
    return degrees * M_PI / 180.0;
}

/**
 * @brief 弧度转角度
 */
inline double radToDeg(double radians) {
    return radians * 180.0 / M_PI;
}

/**
 * @brief 限制值在范围内
 */
template<typename T>
inline T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

/**
 * @brief 线性插值
 */
template<typename T>
inline T lerp(const T& a, const T& b, double t) {
    return a + (b - a) * t;
}

} // namespace math_utils
} // namespace endorobo

