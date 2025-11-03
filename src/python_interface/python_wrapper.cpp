#include "python_interface/python_wrapper.h"
#include "core/logger.h"
#include <numpy/arrayobject.h>

namespace endorobo {

PythonWrapper::PythonWrapper()
    : initialized_(false)
    , numpy_module_(nullptr) {
}

PythonWrapper::~PythonWrapper() {
    if (initialized_) {
        Py_XDECREF(numpy_module_);
        Py_Finalize();
        LOG_INFO("Python interpreter finalized");
    }
}

bool PythonWrapper::initialize(const std::string& python_path) {
    if (initialized_) {
        LOG_WARNING("Python interpreter already initialized");
        return true;
    }
    
    LOG_INFO("Initializing Python interpreter...");
    
    // 设置Python路径
    if (!python_path.empty()) {
        std::wstring wide_path(python_path.begin(), python_path.end());
        Py_SetPath(wide_path.c_str());
    }
    
    // 初始化Python解释器
    Py_Initialize();
    
    if (!Py_IsInitialized()) {
        LOG_ERROR("Failed to initialize Python interpreter");
        return false;
    }
    
    // 初始化NumPy
    if (!initializeNumpy()) {
        LOG_ERROR("Failed to initialize NumPy");
        Py_Finalize();
        return false;
    }
    
    initialized_ = true;
    LOG_INFO("Python interpreter initialized successfully");
    return true;
}

bool PythonWrapper::initializeNumpy() {
    import_array1(false);  // 初始化NumPy C API
    
    numpy_module_ = PyImport_ImportModule("numpy");
    if (!numpy_module_) {
        checkError("Failed to import numpy");
        return false;
    }
    
    LOG_INFO("NumPy module loaded successfully");
    return true;
}

PyObject* PythonWrapper::loadModule(const std::string& module_name) {
    if (!initialized_) {
        LOG_ERROR("Python interpreter not initialized");
        return nullptr;
    }
    
    PyObject* pName = PyUnicode_DecodeFSDefault(module_name.c_str());
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (!pModule) {
        checkError("Failed to load module: " + module_name);
        return nullptr;
    }
    
    LOG_INFO("Module loaded: ", module_name);
    return pModule;
}

PyObject* PythonWrapper::loadFunction(PyObject* module, const std::string& func_name) {
    if (!module) {
        LOG_ERROR("Invalid module pointer");
        return nullptr;
    }
    
    PyObject* pFunc = PyObject_GetAttrString(module, func_name.c_str());
    
    if (!pFunc || !PyCallable_Check(pFunc)) {
        checkError("Failed to load function: " + func_name);
        Py_XDECREF(pFunc);
        return nullptr;
    }
    
    LOG_INFO("Function loaded: ", func_name);
    return pFunc;
}

PyObject* PythonWrapper::matToNumpy(const cv::Mat& mat) {
    if (mat.empty()) {
        LOG_ERROR("Input Mat is empty");
        return nullptr;
    }
    
    // 确保数据连续
    cv::Mat continuous_mat = mat.clone();
    
    npy_intp dims[3] = {continuous_mat.rows, continuous_mat.cols, continuous_mat.channels()};
    int nd = (continuous_mat.channels() > 1) ? 3 : 2;
    
    // 确定NumPy数据类型
    int np_type = NPY_UINT8;
    if (continuous_mat.depth() == CV_32F) {
        np_type = NPY_FLOAT32;
    } else if (continuous_mat.depth() == CV_64F) {
        np_type = NPY_FLOAT64;
    }
    
    // 创建NumPy数组
    PyObject* numpy_array = PyArray_SimpleNewFromData(
        nd, dims, np_type, continuous_mat.data);
    
    if (!numpy_array) {
        checkError("Failed to create NumPy array");
        return nullptr;
    }
    
    // 创建拷贝以避免内存管理问题
    PyObject* numpy_copy = PyArray_NewCopy((PyArrayObject*)numpy_array, NPY_CORDER);
    Py_DECREF(numpy_array);
    
    return numpy_copy;
}

cv::Mat PythonWrapper::numpyToMat(PyObject* numpy_array) {
    if (!numpy_array || !PyArray_Check(numpy_array)) {
        LOG_ERROR("Invalid NumPy array");
        return cv::Mat();
    }
    
    PyArrayObject* np_arr = reinterpret_cast<PyArrayObject*>(numpy_array);
    
    int ndims = PyArray_NDIM(np_arr);
    npy_intp* dims = PyArray_DIMS(np_arr);
    
    // 确定OpenCV数据类型
    int cv_type = CV_8UC1;
    int np_type = PyArray_TYPE(np_arr);
    
    if (np_type == NPY_FLOAT32) {
        cv_type = (ndims == 3) ? CV_32FC(dims[2]) : CV_32FC1;
    } else if (np_type == NPY_FLOAT64) {
        cv_type = (ndims == 3) ? CV_64FC(dims[2]) : CV_64FC1;
    } else if (np_type == NPY_UINT8) {
        cv_type = (ndims == 3) ? CV_8UC(dims[2]) : CV_8UC1;
    }
    
    // 创建OpenCV Mat
    cv::Mat mat;
    if (ndims == 2) {
        mat = cv::Mat(dims[0], dims[1], cv_type, PyArray_DATA(np_arr));
    } else if (ndims == 3) {
        int sizes[3] = {static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2])};
        mat = cv::Mat(ndims, sizes, cv_type, PyArray_DATA(np_arr));
        // 如果是图像，转换为标准的HWC格式
        if (dims[2] <= 4) {
            mat = mat.reshape(dims[2], dims[0]);
        }
    }
    
    // 返回拷贝以避免内存管理问题
    return mat.clone();
}

bool PythonWrapper::checkError(const std::string& context) {
    if (PyErr_Occurred()) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        
        PyObject* pstr = PyObject_Str(pvalue);
        const char* error_msg = PyUnicode_AsUTF8(pstr);
        
        if (!context.empty()) {
            LOG_ERROR(context, ": ", error_msg ? error_msg : "Unknown error");
        } else {
            LOG_ERROR("Python error: ", error_msg ? error_msg : "Unknown error");
        }
        
        Py_XDECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        
        return true;
    }
    return false;
}

} // namespace endorobo

