#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    
    cout << "=== Simple Camera Calibration ===" << endl;
    cout << "OpenCV Version: " << CV_VERSION << endl;
    
    // 棋盘格参数
    Size boardSize(9, 6);  // 内角点数量
    float squareSize = 25.0f;  // 方格大小(mm)
    
    // 标定数据
    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints;
    Size imageSize;
    
    // 生成棋盘格世界坐标
    vector<Point3f> objectCorners;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objectCorners.push_back(Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    
    // 打开相机
    int cameraId = 0;
    if (argc > 1) {
        cameraId = atoi(argv[1]);
    }
    
    VideoCapture cap(cameraId);
    if (!cap.isOpened()) {
        cout << "Cannot open camera " << cameraId << endl;
        return -1;
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    Mat frame;
    int imageCount = 0;
    
    cout << "\nInstructions:" << endl;
    cout << "- Press SPACE to capture image when chessboard is detected" << endl;
    cout << "- Press ESC to exit capture" << endl;
    cout << "- Press C to start calibration (need at least 10 images)" << endl;
    cout << "- Capture 15-20 images from different angles\n" << endl;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        Mat gray, display = frame.clone();
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 检测棋盘格角点
        vector<Point2f> corners2f;
        bool found = findChessboardCorners(gray, boardSize, corners2f,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        
        if (found) {
            // 亚像素精度优化
            cornerSubPix(gray, corners2f, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            
            // 绘制角点
            drawChessboardCorners(display, boardSize, corners2f, found);
            putText(display, "Chessboard Detected - Press SPACE to capture", 
                   Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        } else {
            putText(display, "No Chessboard - Adjust position", 
                   Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }
        
        putText(display, "Images captured: " + to_string(imageCount), 
               Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        
        imshow("Camera Calibration", display);
        
        int key = waitKey(1) & 0xFF;
        if (key == 27) { // ESC
            break;
        } else if (key == 32 && found) { // SPACE
            imagePoints.push_back(corners2f);
            objectPoints.push_back(objectCorners);
            imageCount++;
            
            if (imageSize.width == 0) {
                imageSize = frame.size();
            }
            
            // 保存图像
            string filename = "images/calib_" + to_string(imageCount) + ".jpg";
            imwrite(filename, frame);
            cout << "Captured image " << imageCount << endl;
            
        } else if ((key == 'c' || key == 'C') && imageCount >= 10) {
            break;
        } else if (key == 'c' || key == 'C') {
            cout << "Need at least 10 images, current: " << imageCount << endl;
        }
    }
    
    destroyAllWindows();
    
    if (imageCount < 10) {
        cout << "Not enough images for calibration" << endl;
        return -1;
    }
    
    // 执行标定
    cout << "\nStarting calibration with " << imageCount << " images..." << endl;
    
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvecs, tvecs;
    
    double rmsError = calibrateCamera(objectPoints, imagePoints, imageSize,
                                     cameraMatrix, distCoeffs, rvecs, tvecs);
    
    cout << "Calibration completed!" << endl;
    cout << "RMS reprojection error: " << rmsError << " pixels" << endl;
    
    // 保存结果
    FileStorage fs("camera_calibration.xml", FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "rms_reprojection_error" << rmsError;
    fs.release();
    
    // 输出结果
    cout << "\n=== Calibration Results ===" << endl;
    cout << "Image size: " << imageSize.width << " x " << imageSize.height << endl;
    cout << "Camera matrix:" << endl << cameraMatrix << endl;
    cout << "Distortion coefficients:" << endl << distCoeffs.t() << endl;
    
    // 提取参数
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);
    
    double k1 = distCoeffs.at<double>(0);
    double k2 = distCoeffs.at<double>(1);
    double p1 = distCoeffs.at<double>(2);
    double p2 = distCoeffs.at<double>(3);
    double k3 = distCoeffs.at<double>(4);
    
    cout << "\n=== Update config/camera_config.yaml ===" << endl;
    cout << "intrinsics:" << endl;
    cout << "  fx: " << fx << endl;
    cout << "  fy: " << fy << endl;
    cout << "  cx: " << cx << endl;
    cout << "  cy: " << cy << endl;
    cout << "distortion:" << endl;
    cout << "  k1: " << k1 << endl;
    cout << "  k2: " << k2 << endl;
    cout << "  k3: " << k3 << endl;
    cout << "  p1: " << p1 << endl;
    cout << "  p2: " << p2 << endl;
    
    cout << "\nCalibration results saved to camera_calibration.xml" << endl;
    cout << "Please update the config file with the above values." << endl;
    
    return 0;
}
