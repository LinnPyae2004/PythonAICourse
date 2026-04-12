#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    cv::Mat image = cv::Mat::zeros(600, 600, CV_8UC3); // Create black image
    cv::putText(image, "Let's Sleep!!!", cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Test Window", image);
    cv::waitKey(0);
    return 0;
}