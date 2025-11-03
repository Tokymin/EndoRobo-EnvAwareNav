#include "python_interface/depth_estimator.h"
#include "core/logger.h"

namespace endorobo {

DepthEstimator::DepthEstimator(const PythonModelConfig& config,
                               const CameraConfig& camera_config)
    : config_(config)
    , camera_config_(camera_config)
    , model_module_(nullptr)
    , predict_func_(nullptr)
    , initialized_(false) {
}

DepthEstimator::~DepthEstimator() {
    if (predict_func_) {
        Py_DECREF(predict_func_);
    }
    if (model_module_) {
        Py_DECREF(model_module_);
    }
}

bool DepthEstimator::initialize(std::shared_ptr<PythonWrapper> python_wrapper) {
    LOG_INFO("Initializing depth estimator...");
    
    if (!python_wrapper || !python_wrapper->isInitialized()) {
        LOG_ERROR("Invalid or uninitialized Python wrapper");
        return false;
    }
    
    python_wrapper_ = python_wrapper;
    
    // 加载深度估计模型模块
    // 这里假设Python模块名为 "depth_model"
    model_module_ = python_wrapper_->loadModule("depth_model");
    if (!model_module_) {
        LOG_ERROR("Failed to load depth estimation model module");
        return false;
    }
    
    // 加载预测函数
    predict_func_ = python_wrapper_->loadFunction(model_module_, "predict_depth");
    if (!predict_func_) {
        LOG_ERROR("Failed to load predict_depth function");
        return false;
    }
    
    initialized_ = true;
    LOG_INFO("Depth estimator initialized successfully");
    LOG_INFO("Model path: ", config_.model_path);
    LOG_INFO("Input size: ", config_.input_size.width, "x", config_.input_size.height);
    LOG_INFO("Depth range: [", config_.min_depth, ", ", config_.max_depth, "]");
    
    return true;
}

cv::Mat DepthEstimator::preprocessImage(const cv::Mat& image) {
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

cv::Mat DepthEstimator::postprocessDepth(const cv::Mat& depth_map) {
    cv::Mat processed;
    
    // 将深度值缩放到实际范围
    depth_map.convertTo(processed, CV_32F);
    processed = processed * (config_.max_depth - config_.min_depth) + config_.min_depth;
    
    // 中值滤波去噪
    cv::medianBlur(processed, processed, 5);
    
    return processed;
}

bool DepthEstimator::estimateDepth(const cv::Mat& frame, DepthEstimation& depth) {
    if (!initialized_) {
        LOG_ERROR("Depth estimator not initialized");
        return false;
    }
    
    if (frame.empty()) {
        LOG_ERROR("Input frame is empty");
        return false;
    }
    
    // 预处理图像
    cv::Mat processed = preprocessImage(frame);
    
    // 转换为NumPy数组
    PyObject* input_numpy = python_wrapper_->matToNumpy(processed);
    if (!input_numpy) {
        LOG_ERROR("Failed to convert frame to NumPy array");
        return false;
    }
    
    // 调用Python函数
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, input_numpy);
    
    PyObject* pResult = PyObject_CallObject(predict_func_, pArgs);
    Py_DECREF(pArgs);
    
    if (!pResult) {
        python_wrapper_->checkError("Failed to call predict_depth");
        return false;
    }
    
    // 转换结果为OpenCV Mat
    cv::Mat depth_map_raw = python_wrapper_->numpyToMat(pResult);
    Py_DECREF(pResult);
    
    if (depth_map_raw.empty()) {
        LOG_ERROR("Failed to convert depth result to Mat");
        return false;
    }
    
    // 后处理深度图
    depth.depth_map = postprocessDepth(depth_map_raw);
    
    // 调整到原始图像大小
    if (depth.depth_map.size() != frame.size()) {
        cv::resize(depth.depth_map, depth.depth_map, frame.size());
    }
    
    // 提取特征点
    extractFeatures(depth.depth_map, frame, depth.features);
    
    depth.valid = true;
    
    return true;
}

bool DepthEstimator::extractFeatures(const cv::Mat& depth_map,
                                    const cv::Mat& rgb_image,
                                    std::vector<FeaturePoint>& features,
                                    int num_features) {
    features.clear();
    
    // 使用GFTT（Good Features To Track）检测角点
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(rgb_image, corners, num_features, 0.01, 10);
    
    // 为每个角点添加深度信息
    for (const auto& corner : corners) {
        int x = static_cast<int>(corner.x);
        int y = static_cast<int>(corner.y);
        
        // 边界检查
        if (x < 0 || x >= depth_map.cols || y < 0 || y >= depth_map.rows) {
            continue;
        }
        
        float depth_value = depth_map.at<float>(y, x);
        
        // 深度有效性检查
        if (depth_value < config_.min_depth || depth_value > config_.max_depth) {
            continue;
        }
        
        FeaturePoint fp;
        fp.pixel = corner;
        fp.depth = depth_value;
        fp.point_3d = pixelToPoint3D(corner, depth_value);
        fp.confidence = 1.0f;  // 可以根据深度图的质量调整
        
        features.push_back(fp);
    }
    
    LOG_DEBUG("Extracted ", features.size(), " features from ", corners.size(), " corners");
    
    return !features.empty();
}

Eigen::Vector3d DepthEstimator::pixelToPoint3D(const cv::Point2f& pixel, float depth) {
    // 使用相机内参将2D像素点反投影到3D空间
    double fx = camera_config_.fx;
    double fy = camera_config_.fy;
    double cx = camera_config_.cx;
    double cy = camera_config_.cy;
    
    double x = (pixel.x - cx) * depth / fx;
    double y = (pixel.y - cy) * depth / fy;
    double z = depth;
    
    return Eigen::Vector3d(x, y, z);
}

} // namespace endorobo

