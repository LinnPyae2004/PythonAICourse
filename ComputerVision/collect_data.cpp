// acutally this code is doing the same thing as collectd_data.py, I am just bored and wrote this in c++

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

void cleanData(const std::string &directoryPath);

int main()
{
    // change these based on the folder you want to save
    std::string outputFolder = "Open_Palm";
    std::string outputDir = "data/HandGesture/raw_staging/" + outputFolder;

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }

    cleanData(outputDir);

    // camera setup
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    int frameInterval = 10;
    int count = 0;
    int savedCount = 0;
    cv::Mat frame;

    std::cout << "Starting capture, move your hand around! Press 'q' to stop." << std::endl;

    while (true)
    {
        if (!cap.read(frame))
            break;

        // flip to get mirro image
        cv::flip(frame, frame, 1);

        // save the image every 10th frame
        if (count % frameInterval == 0)
        {
            // timestamp of the raw images
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

            // reconstruct the file name with an extension
            std::string filePath = outputDir + "/gesture_" + std::to_string(millis) + ".jpg";

            try {
                if (!cv::imwrite(filePath, frame)) {
                    std::cerr << "Error: imwrite returned false for " << filePath << std::endl;
                }
            } catch (const cv::Exception& e) {
                std::cerr << "OpenCV imwrite failed: " << e.what() << std::endl;
            }
            savedCount++;

            cv::circle(frame, cv::Point(30, 30), 10, cv::Scalar(0, 255, 0), -1); // flash circle when saving, I don't think I even need this anyway
            std::cout << "Image Number: " << savedCount << "\r" << std::flush;
        }

        cv::imshow("Capturing Data - Press 'q' to Quit", frame);

        count++;

        // press 'q' on keyboard to exit
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    std::cout << "\nDone! Saved " << savedCount << " images to " << outputDir << std::endl;

    return 0;
}

// clean old image files
void cleanData(const std::string &directoryPath)
{
    std::cout << "Cleaning old image files in: " << directoryPath << std::endl;

    try
    {
        if (fs::exists(directoryPath) && fs::is_directory(directoryPath))
        {
            for (const auto &entry : fs::directory_iterator(directoryPath))
            {
                if (fs::is_regular_file(entry.path()))
                {
                    fs::remove(entry.path());
                }
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Error during cleanup: " << e.what() << std::endl;
    }
}