#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

/**
 * @brief 相机标定程序
 * 
 * 使用棋盘格标定相机内参和畸变系数
 * 
 * 棋盘格要求：
 * - 尺寸：9x6 内角点（10x7 方格）
 * - 方格大小：25mm x 25mm
 * - 打印在A4纸上，确保平整
 * - 建议贴在硬质板上防止弯曲
 */

class CameraCalibration {
private:
    // 棋盘格参数
    cv::Size board_size_;           // 内角点数量 (9x6)
    float square_size_;             // 方格实际大小 (mm)
    
    // 标定数据
    std::vector<std::vector<cv::Point2f>> image_points_;    // 图像角点
    std::vector<std::vector<cv::Point3f>> object_points_;   // 世界坐标角点
    cv::Size image_size_;                                   // 图像尺寸
    
    // 标定结果
    cv::Mat camera_matrix_;         // 相机内参矩阵
    cv::Mat dist_coeffs_;          // 畸变系数
    std::vector<cv::Mat> rvecs_;   // 旋转向量
    std::vector<cv::Mat> tvecs_;   // 平移向量
    double rms_error_;             // 重投影误差

public:
    CameraCalibration(cv::Size board_size = cv::Size(9, 6), float square_size = 25.0f)
        : board_size_(board_size), square_size_(square_size), rms_error_(0.0) {
        
        // 初始化相机矩阵和畸变系数
        camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
        dist_coeffs_ = cv::Mat::zeros(8, 1, CV_64F);
    }
    
    /**
     * @brief 生成棋盘格的世界坐标点
     */
    std::vector<cv::Point3f> generateObjectPoints() {
        std::vector<cv::Point3f> corners;
        for (int i = 0; i < board_size_.height; i++) {
            for (int j = 0; j < board_size_.width; j++) {
                corners.push_back(cv::Point3f(j * square_size_, i * square_size_, 0));
            }
        }
        return corners;
    }
    
    /**
     * @brief 从图像中检测棋盘格角点
     */
    bool detectChessboard(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        // 检测棋盘格角点
        bool found = cv::findChessboardCorners(gray, board_size_, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        
        if (found) {
            // 亚像素精度优化
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        }
        
        return found;
    }
    
    /**
     * @brief 添加标定图像
     */
    bool addCalibrationImage(const cv::Mat& image) {
        std::vector<cv::Point2f> corners;
        
        if (detectChessboard(image, corners)) {
            image_points_.push_back(corners);
            object_points_.push_back(generateObjectPoints());
            
            if (image_size_.width == 0) {
                image_size_ = image.size();
            }
            
            std::cout << "成功添加第 " << image_points_.size() << " 张标定图像" << std::endl;
            return true;
        } else {
            std::cout << "未检测到棋盘格角点" << std::endl;
            return false;
        }
    }
    
    /**
     * @brief 执行相机标定
     */
    bool calibrate() {
        if (image_points_.size() < 10) {
            std::cout << "标定图像数量不足，至少需要10张，当前: " << image_points_.size() << std::endl;
            return false;
        }
        
        std::cout << "开始相机标定，使用 " << image_points_.size() << " 张图像..." << std::endl;
        
        // 执行标定
        rms_error_ = cv::calibrateCamera(
            object_points_, image_points_, image_size_,
            camera_matrix_, dist_coeffs_, rvecs_, tvecs_,
            cv::CALIB_FIX_PRINCIPAL_POINT);
        
        std::cout << "标定完成！RMS重投影误差: " << rms_error_ << " 像素" << std::endl;
        
        return rms_error_ < 1.0;  // 误差小于1像素认为标定成功
    }
    
    /**
     * @brief 保存标定结果到YAML文件
     */
    void saveCalibrationResults(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        
        fs << "image_width" << image_size_.width;
        fs << "image_height" << image_size_.height;
        fs << "board_width" << board_size_.width;
        fs << "board_height" << board_size_.height;
        fs << "square_size" << square_size_;
        fs << "camera_matrix" << camera_matrix_;
        fs << "distortion_coefficients" << dist_coeffs_;
        fs << "rms_reprojection_error" << rms_error_;
        
        fs.release();
        
        std::cout << "标定结果已保存到: " << filename << std::endl;
        printCalibrationResults();
    }
    
    /**
     * @brief 更新项目配置文件
     */
    void updateProjectConfig(const std::string& config_file) {
        // 读取现有配置
        cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
        if (!fs_read.isOpened()) {
            std::cout << "无法打开配置文件: " << config_file << std::endl;
            return;
        }
        
        // 提取内参
        double fx = camera_matrix_.at<double>(0, 0);
        double fy = camera_matrix_.at<double>(1, 1);
        double cx = camera_matrix_.at<double>(0, 2);
        double cy = camera_matrix_.at<double>(1, 2);
        
        double k1 = dist_coeffs_.at<double>(0);
        double k2 = dist_coeffs_.at<double>(1);
        double p1 = dist_coeffs_.at<double>(2);
        double p2 = dist_coeffs_.at<double>(3);
        double k3 = dist_coeffs_.at<double>(4);
        
        fs_read.release();
        
        // 输出更新建议
        std::cout << "\n=== 请手动更新 config/camera_config.yaml ===" << std::endl;
        std::cout << "intrinsics:" << std::endl;
        std::cout << "  fx: " << fx << std::endl;
        std::cout << "  fy: " << fy << std::endl;
        std::cout << "  cx: " << cx << std::endl;
        std::cout << "  cy: " << cy << std::endl;
        std::cout << "distortion:" << std::endl;
        std::cout << "  k1: " << k1 << std::endl;
        std::cout << "  k2: " << k2 << std::endl;
        std::cout << "  k3: " << k3 << std::endl;
        std::cout << "  p1: " << p1 << std::endl;
        std::cout << "  p2: " << p2 << std::endl;
    }
    
    /**
     * @brief 打印标定结果
     */
    void printCalibrationResults() {
        std::cout << "\n=== 相机标定结果 ===" << std::endl;
        std::cout << "图像尺寸: " << image_size_.width << " x " << image_size_.height << std::endl;
        std::cout << "标定图像数量: " << image_points_.size() << std::endl;
        std::cout << "RMS重投影误差: " << rms_error_ << " 像素" << std::endl;
        
        std::cout << "\n相机内参矩阵:" << std::endl;
        std::cout << camera_matrix_ << std::endl;
        
        std::cout << "\n畸变系数:" << std::endl;
        std::cout << "k1=" << dist_coeffs_.at<double>(0) << ", ";
        std::cout << "k2=" << dist_coeffs_.at<double>(1) << ", ";
        std::cout << "p1=" << dist_coeffs_.at<double>(2) << ", ";
        std::cout << "p2=" << dist_coeffs_.at<double>(3) << ", ";
        std::cout << "k3=" << dist_coeffs_.at<double>(4) << std::endl;
    }
    
    /**
     * @brief 测试标定结果（去畸变）
     */
    void testUndistortion(const cv::Mat& image) {
        cv::Mat undistorted;
        cv::undistort(image, undistorted, camera_matrix_, dist_coeffs_);
        
        cv::imshow("原始图像", image);
        cv::imshow("去畸变图像", undistorted);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
};

/**
 * @brief 实时采集标定图像
 */
void captureCalibrationImages(CameraCalibration& calibrator, int camera_id = 0) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cout << "无法打开相机 " << camera_id << std::endl;
        return;
    }
    
    // 设置相机参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    cv::Mat frame;
    int image_count = 0;
    
    std::cout << "\n=== 实时采集标定图像 ===" << std::endl;
    std::cout << "操作说明:" << std::endl;
    std::cout << "- 按 SPACE 键采集当前图像" << std::endl;
    std::cout << "- 按 ESC 键退出采集" << std::endl;
    std::cout << "- 按 C 键开始标定" << std::endl;
    std::cout << "建议采集15-20张不同角度和位置的图像\n" << std::endl;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // 检测棋盘格并绘制
        std::vector<cv::Point2f> corners;
        cv::Mat display_frame = frame.clone();
        
        if (calibrator.detectChessboard(frame, corners)) {
            cv::drawChessboardCorners(display_frame, cv::Size(9, 6), corners, true);
            cv::putText(display_frame, "Chessboard Detected - Press SPACE to capture", 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(display_frame, "No Chessboard - Adjust position", 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        
        cv::putText(display_frame, "Images: " + std::to_string(image_count), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Camera Calibration", display_frame);
        
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) { // ESC
            break;
        } else if (key == 32) { // SPACE
            if (calibrator.detectChessboard(frame, corners)) {
                if (calibrator.addCalibrationImage(frame)) {
                    image_count++;
                    // 保存图像
                    std::string filename = "calibration/images/calib_" + 
                                         std::to_string(image_count) + ".jpg";
                    cv::imwrite(filename, frame);
                }
            }
        } else if (key == 'c' || key == 'C') {
            if (image_count >= 10) {
                break;
            } else {
                std::cout << "图像数量不足，至少需要10张，当前: " << image_count << std::endl;
            }
        }
    }
    
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    std::cout << "=== OpenCV 相机标定程序 ===" << std::endl;
    std::cout << "版本: OpenCV " << CV_VERSION << std::endl;
    
    // 创建标定器
    CameraCalibration calibrator(cv::Size(9, 6), 25.0f);
    
    int camera_id = 0;
    if (argc > 1) {
        camera_id = std::atoi(argv[1]);
    }
    
    // 实时采集标定图像
    captureCalibrationImages(calibrator, camera_id);
    
    // 执行标定
    if (calibrator.calibrate()) {
        // 保存结果
        calibrator.saveCalibrationResults("calibration/camera_calibration.xml");
        calibrator.updateProjectConfig("config/camera_config.yaml");
        
        std::cout << "\n标定成功！请根据上述输出更新 config/camera_config.yaml 文件" << std::endl;
    } else {
        std::cout << "标定失败，请检查图像质量和数量" << std::endl;
    }
    
    return 0;
}
