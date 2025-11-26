#include "python_interface/depth_estimator.h"
#include "core/logger.h"
#include <filesystem>

namespace endorobo {

DepthEstimator::DepthEstimator(const PythonModelConfig& config,
                               const CameraConfig& camera_config)
    : config_(config)
    , camera_config_(camera_config)
    , model_module_(nullptr)
    , predict_func_(nullptr)
    , depth_model_(nullptr)
    , initialized_(false) {
}

DepthEstimator::~DepthEstimator() {
    if (Py_IsInitialized()) {
        ScopedGILLock gil;
        Py_XDECREF(predict_func_);
        Py_XDECREF(model_module_);
        Py_XDECREF(depth_model_);
    }
}

bool DepthEstimator::initialize(std::shared_ptr<PythonWrapper> python_wrapper) {
    LOG_INFO("Initializing Depth Anything V2...");
    
    if (!python_wrapper || !python_wrapper->isInitialized()) {
        LOG_ERROR("Invalid or uninitialized Python wrapper");
        return false;
    }
    
    python_wrapper_ = python_wrapper;
    
    ScopedGILLock gil;
    
    try {
        // 导入Depth Anything V2模块
        LOG_INFO("Attempting to import: depth_model_dav2");
        
        // 先检查sys.path
        PyObject* sys = PyImport_ImportModule("sys");
        if (sys) {
            PyObject* sys_path = PyObject_GetAttrString(sys, "path");
            if (sys_path && PyList_Check(sys_path)) {
                Py_ssize_t size = PyList_Size(sys_path);
                LOG_INFO("Python sys.path has " + std::to_string(size) + " entries:");
                for (Py_ssize_t i = 0; i < std::min(size, (Py_ssize_t)5); i++) {
                    PyObject* item = PyList_GetItem(sys_path, i);
                    if (item) {
                        PyObject* str = PyObject_Str(item);
                        if (str) {
                            const char* path_str = PyUnicode_AsUTF8(str);
                            if (path_str) {
                                LOG_INFO("  [" + std::to_string(i) + "] " + std::string(path_str));
                            }
                            Py_DECREF(str);
                        }
                    }
                }
                Py_DECREF(sys_path);
            }
            Py_DECREF(sys);
        }
        
        PyObject* module_name = PyUnicode_FromString("depth_model_dav2");
        model_module_ = PyImport_Import(module_name);
        Py_XDECREF(module_name);
        
        if (!model_module_) {
            LOG_ERROR("Failed to import depth_model_dav2");
            
            // Get detailed error information
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            
            if (pvalue) {
                PyObject* str_exc = PyObject_Str(pvalue);
                if (str_exc) {
                    const char* err_msg = PyUnicode_AsUTF8(str_exc);
                    if (err_msg) {
                        LOG_ERROR("Python error: " + std::string(err_msg));
                    }
                    Py_DECREF(str_exc);
                }
            }
            
            // Restore and print full traceback
            PyErr_Restore(ptype, pvalue, ptraceback);
            PyErr_Print();
            
            return false;
        }
        
        // Get DepthModel class
        PyObject* depth_class = PyObject_GetAttrString(model_module_, "DepthModel");
        
        if (!depth_class || !PyCallable_Check(depth_class)) {
            LOG_ERROR("DepthModel class not found or not callable");
            Py_XDECREF(depth_class);
            return false;
        }
        
        // Create model instance (using Small model for fastest speed)
        PyObject* args = PyTuple_New(0);
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "model_size", PyUnicode_FromString("vits"));
        PyDict_SetItemString(kwargs, "device", PyUnicode_FromString("cuda"));  // Use CUDA for GPU acceleration
        
        depth_model_ = PyObject_Call(depth_class, args, kwargs);
        Py_XDECREF(args);
        Py_XDECREF(kwargs);
        Py_XDECREF(depth_class);
        
        if (!depth_model_) {
            LOG_ERROR("Failed to create DepthModel instance");
            
            // Get detailed Python error information
            PyObject *ptype2, *pvalue2, *ptraceback2;
            PyErr_Fetch(&ptype2, &pvalue2, &ptraceback2);
            
            if (pvalue2) {
                PyObject* str_exc = PyObject_Str(pvalue2);
                if (str_exc) {
                    const char* err_msg = PyUnicode_AsUTF8(str_exc);
                    if (err_msg) {
                        LOG_ERROR("Python error details: " + std::string(err_msg));
                    }
                    Py_DECREF(str_exc);
                }
                
                // Print traceback if available
                if (ptraceback2) {
                    PyObject* traceback_module = PyImport_ImportModule("traceback");
                    if (traceback_module) {
                        PyObject* format_tb = PyObject_GetAttrString(traceback_module, "format_tb");
                        if (format_tb && PyCallable_Check(format_tb)) {
                            PyObject* tb_list = PyObject_CallFunctionObjArgs(format_tb, ptraceback2, NULL);
                            if (tb_list && PyList_Check(tb_list)) {
                                Py_ssize_t size = PyList_Size(tb_list);
                                for (Py_ssize_t i = 0; i < size; i++) {
                                    PyObject* item = PyList_GetItem(tb_list, i);
                                    if (item) {
                                        const char* tb_str = PyUnicode_AsUTF8(item);
                                        if (tb_str) {
                                            LOG_ERROR("Traceback: " + std::string(tb_str));
                                        }
                                    }
                                }
                                Py_DECREF(tb_list);
                            }
                            Py_XDECREF(format_tb);
                        }
                        Py_DECREF(traceback_module);
                    }
                }
            }
            
            // Restore and print error
            PyErr_Restore(ptype2, pvalue2, ptraceback2);
            PyErr_Print();
            
            return false;
        }
        
        // Get estimate_depth method
        predict_func_ = PyObject_GetAttrString(depth_model_, "estimate_depth");
        if (!predict_func_ || !PyCallable_Check(predict_func_)) {
            LOG_ERROR("estimate_depth method not found");
            return false;
        }
        
        initialized_ = true;
        LOG_INFO("Depth Anything V2 initialized successfully!");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception in depth estimator initialization: ", e.what());
        return false;
    }
}

bool DepthEstimator::estimateDepth(const cv::Mat& image, DepthEstimation& result) {
    if (!initialized_ || !python_wrapper_ || !depth_model_ || !predict_func_) {
        return false;
    }
    
    try {
        // 转换输入图像为NumPy数组
        PyObject* np_image = python_wrapper_->matToNumpy(image);
        if (!np_image) {
            LOG_ERROR("Failed to convert image to numpy");
            return false;
        }
        
        // 调用estimate_depth函数
        PyObject* args = PyTuple_Pack(1, np_image);
        PyObject* depth_result = PyObject_CallObject(predict_func_, args);
        Py_XDECREF(args);
        Py_XDECREF(np_image);
        
        if (!depth_result) {
            PyErr_Print();
            return false;
        }
        
        // 转换结果为cv::Mat
        cv::Mat depth_map = python_wrapper_->numpyToMat(depth_result);
        Py_XDECREF(depth_result);
        
        if (depth_map.empty()) {
            return false;
        }
        
        // 调试：记录原始深度图统计信息并保存前几帧原始图
        double min_val = 0.0, max_val = 0.0;
        cv::minMaxLoc(depth_map, &min_val, &max_val);
        LOG_INFO("DepthEstimator: raw depth_map type=", depth_map.type(),
                 " size=", depth_map.cols, "x", depth_map.rows,
                 " min=", min_val, " max=", max_val);

        static int debug_depth_dump_count = 0;
        if (debug_depth_dump_count < 5) {
            std::error_code ec;
            std::filesystem::create_directories("output/debug", ec);
            std::string filename = "output/debug/depth_raw_" +
                                   std::to_string(debug_depth_dump_count) + ".png";
            if (cv::imwrite(filename, depth_map)) {
                LOG_INFO("DepthEstimator: saved raw depth to ", filename);
            } else {
                LOG_WARNING("DepthEstimator: failed to save raw depth to ", filename);
            }
            debug_depth_dump_count++;
        }

        // 转换为以米为单位的32位浮点深度图
        const double alpha = (config_.max_depth - config_.min_depth) / 255.0;
        cv::Mat depth_float;
        depth_map.convertTo(depth_float, CV_32F, alpha, config_.min_depth);

        double float_min = 0.0, float_max = 0.0;
        cv::minMaxLoc(depth_float, &float_min, &float_max);
        LOG_INFO("DepthEstimator: converted depth type=", depth_float.type(),
                 " size=", depth_float.cols, "x", depth_float.rows,
                 " min=", float_min, " max=", float_max);

        // 填充结果
        result.depth_map = depth_float.clone();
        result.valid = true;
        
        // 提取深度特征点（简单采样）
        result.feature_points.clear();
        const int step = 20;  // 每20个像素采样一个点
        for (int y = 0; y < depth_float.rows; y += step) {
            for (int x = 0; x < depth_float.cols; x += step) {
                float depth = depth_float.at<float>(y, x);
                result.feature_points.push_back(cv::Point3f(static_cast<float>(x),
                                                            static_cast<float>(y),
                                                            depth));
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception in depth estimation: ", e.what());
        return false;
    }
}

bool DepthEstimator::isInitialized() const {
    return initialized_;
}

} // namespace endorobo

