#pragma once

#include <Python.h>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace endorobo {

/**
 * @brief Python解释器包装类
 * 负责管理Python解释器的生命周期和基本操作
 */
class PythonWrapper {
public:
    /**
     * @brief 构造函数
     */
    PythonWrapper();
    
    /**
     * @brief 析构函数
     */
    ~PythonWrapper();
    
    /**
     * @brief 初始化Python解释器
     * @param python_path Python模块搜索路径
     * @return 是否初始化成功
     */
    bool initialize(const std::string& python_path = "");
    
    /**
     * @brief 加载Python模块
     * @param module_name 模块名称
     * @return Python模块对象指针
     */
    PyObject* loadModule(const std::string& module_name);
    
    /**
     * @brief 加载Python函数
     * @param module Python模块对象
     * @param func_name 函数名称
     * @return Python函数对象指针
     */
    PyObject* loadFunction(PyObject* module, const std::string& func_name);
    
    /**
     * @brief 将OpenCV Mat转换为NumPy数组
     * @param mat OpenCV矩阵
     * @return NumPy数组对象
     */
    PyObject* matToNumpy(const cv::Mat& mat);
    
    /**
     * @brief 将NumPy数组转换为OpenCV Mat
     * @param numpy_array NumPy数组对象
     * @return OpenCV矩阵
     */
    cv::Mat numpyToMat(PyObject* numpy_array);
    
    /**
     * @brief 检查Python错误
     * @param context 错误上下文描述
     * @return 是否有错误
     */
    bool checkError(const std::string& context = "");
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_;
    PyObject* numpy_module_;
    
    /**
     * @brief 初始化NumPy
     */
    bool initializeNumpy();
};

} // namespace endorobo

