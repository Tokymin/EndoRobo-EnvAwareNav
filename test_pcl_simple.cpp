// Simple PCL test - just check if headers compile and link
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int main() {
    std::cout << "PCL Simple Test" << std::endl;
    std::cout << "================" << std::endl;
    
    // Just create a point cloud object - no operations
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = 10;
    cloud.height = 1;
    cloud.is_dense = false;
    
    std::cout << "[OK] PCL headers compiled successfully" << std::endl;
    std::cout << "[OK] PCL PointCloud type is available" << std::endl;
    std::cout << "[OK] Cloud created with width: " << cloud.width << std::endl;
    
    std::cout << std::endl << "PCL Library Test: PASSED" << std::endl;
    return 0;
}


