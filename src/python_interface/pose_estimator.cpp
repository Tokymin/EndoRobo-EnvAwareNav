#include "python_interface/pose_estimator.h"
#include "core/logger.h"

namespace endorobo {

PoseEstimator::PoseEstimator(const PythonModelConfig& config)
    : config_(config)
    , model_module_(nullptr)
    , predict_func_(nullptr)
    , initialized_(false) {
}

PoseEstimator::~PoseEstimator() {
    if (Py_IsInitialized()) {
        ScopedGILLock gil;
        Py_XDECREF(predict_func_);
        Py_XDECREF(model_module_);
    }
}

bool PoseEstimator::initialize(std::shared_ptr<PythonWrapper> python_wrapper) {
    LOG_INFO("Initializing pose estimator...");
    
    if (!python_wrapper || !python_wrapper->isInitialized()) {
        LOG_ERROR("Invalid or uninitialized Python wrapper");
        return false;
    }
    
    python_wrapper_ = python_wrapper;
    ScopedGILLock gil;
    
    // 加载位姿估计模型模块
    // 这里假设Python模块名为 "pose_model"
    model_module_ = python_wrapper_->loadModule("pose_model");
    if (!model_module_) {
        LOG_ERROR("Failed to load pose estimation model module");
        return false;
    }
    
    // 加载预测函数
    predict_func_ = python_wrapper_->loadFunction(model_module_, "predict_pose");
    if (!predict_func_) {
        LOG_ERROR("Failed to load predict_pose function");
        return false;
    }
    
    initialized_ = true;
    LOG_INFO("Pose estimator initialized successfully");
    LOG_INFO("Model path: ", config_.model_path);
    LOG_INFO("Input size: ", config_.input_size.width, "x", config_.input_size.height);
    
    return true;
}

cv::Mat PoseEstimator::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // 调整大小
    cv::resize(image, processed, config_.input_size);
    
    // 转换为RGB（如果是BGR）
    if (image.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    // 归一化到[0, 1]
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    
    return processed;
}

bool PoseEstimator::estimatePose(const cv::Mat& current_frame,
                                const cv::Mat& previous_frame,
                                PoseEstimation& pose) {
    if (!initialized_) {
        LOG_ERROR("Pose estimator not initialized");
        return false;
    }
    
    if (current_frame.empty()) {
        LOG_ERROR("Current frame is empty");
        return false;
    }
    
    // 预处理图像
    cv::Mat processed_current = preprocessImage(current_frame);
    cv::Mat processed_previous;
    if (!previous_frame.empty()) {
        processed_previous = preprocessImage(previous_frame);
    }
    
    ScopedGILLock gil;
    
    PyObject* pArgs = PyTuple_New(previous_frame.empty() ? 1 : 2);
    
    // 当前帧
    PyObject* current_numpy = python_wrapper_->matToNumpy(processed_current);
    PyTuple_SetItem(pArgs, 0, current_numpy);
    
    // 前一帧（如果有）
    if (!previous_frame.empty()) {
        PyObject* previous_numpy = python_wrapper_->matToNumpy(processed_previous);
        PyTuple_SetItem(pArgs, 1, previous_numpy);
    }
    
    // 调用Python函数
    PyObject* pResult = PyObject_CallObject(predict_func_, pArgs);
    Py_DECREF(pArgs);
    
    if (!pResult) {
        python_wrapper_->checkError("Failed to call predict_pose");
        return false;
    }
    
    // 解析结果
    bool success = parsePoseResult(pResult, pose);
    Py_DECREF(pResult);
    
    return success;
}

bool PoseEstimator::estimateAbsolutePose(const cv::Mat& frame, PoseEstimation& pose) {
    cv::Mat empty_frame;
    return estimatePose(frame, empty_frame, pose);
}

bool PoseEstimator::parsePoseResult(PyObject* result, PoseEstimation& pose) {
    // 假设Python返回的是一个字典，包含：
    // {
    //   'translation': [x, y, z],
    //   'rotation': [qw, qx, qy, qz] or rotation matrix,
    //   'confidence': float
    // }
    
    if (!PyDict_Check(result)) {
        LOG_ERROR("Expected dictionary result from pose estimation");
        return false;
    }
    
    // 提取平移向量
    PyObject* translation = PyDict_GetItemString(result, "translation");
    if (translation && PyList_Check(translation) && PyList_Size(translation) == 3) {
        pose.translation.x() = PyFloat_AsDouble(PyList_GetItem(translation, 0));
        pose.translation.y() = PyFloat_AsDouble(PyList_GetItem(translation, 1));
        pose.translation.z() = PyFloat_AsDouble(PyList_GetItem(translation, 2));
    }
    
    // 提取旋转（假设为四元数 [w, x, y, z]）
    PyObject* rotation = PyDict_GetItemString(result, "rotation");
    if (rotation && PyList_Check(rotation) && PyList_Size(rotation) == 4) {
        double w = PyFloat_AsDouble(PyList_GetItem(rotation, 0));
        double x = PyFloat_AsDouble(PyList_GetItem(rotation, 1));
        double y = PyFloat_AsDouble(PyList_GetItem(rotation, 2));
        double z = PyFloat_AsDouble(PyList_GetItem(rotation, 3));
        pose.rotation = Eigen::Quaterniond(w, x, y, z);
    }
    
    // 提取置信度
    PyObject* confidence = PyDict_GetItemString(result, "confidence");
    if (confidence) {
        pose.confidence = PyFloat_AsDouble(confidence);
    }
    
    // 构建变换矩阵
    pose.transformation = Eigen::Matrix4d::Identity();
    pose.transformation.block<3, 3>(0, 0) = pose.rotation.toRotationMatrix();
    pose.transformation.block<3, 1>(0, 3) = pose.translation;
    
    pose.valid = (pose.confidence > 0.5);  // 简单的阈值判断
    
    return true;
}

} // namespace endorobo

