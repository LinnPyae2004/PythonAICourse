// doing same thing as detect_hand.py, it's just that I love c++ more lol

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <vector>

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

int main() {
    // load the model (.onnx file)
    std::string modelPath = "data/HandGesture/runs/detect/hand_gesture_v1/weights/best.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        std::cerr << "CUDA not available; using CPU backend\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    // Optional diagnostic:
    std::cout << cv::getBuildInformation() << std::endl;

    cv::VideoCapture cap(0);
    cv::Mat frame;

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    while (cap.read(frame))
    {
        cv::flip(frame, frame, 1);

        // pre-process the frame for YOLO
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
        if (outNames.empty()) {
            net.forward(outputs); // compute all outputs
        } else {
            net.forward(outputs, outNames);
        }

        cv::Mat res = outputs[0].reshape(1, outputs[0].size[1]);
        cv::transpose(res, res);

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int i = 0; i < res.rows; ++i)
        {
            cv::Mat row = res.row(i);
            cv::Mat scores = row.colRange(4, row.cols);
            cv::Point class_id_point;
            double score;
            cv::minMaxLoc(scores, 0, &score, 0, &class_id_point);

            if (score > 0.7)
            { // Your conf=0.7 threshold
                float cx = row.at<float>(0);
                float cy = row.at<float>(1);
                float w = row.at<float>(2);
                float h = row.at<float>(3);

                // Scale boxes back to original frame size
                int left = static_cast<int>((cx - 0.5 * w) * frame.cols / 640);
                int top = static_cast<int>((cy - 0.5 * h) * frame.rows / 640);
                int width = static_cast<int>(w * frame.cols / 640);
                int height = static_cast<int>(h * frame.rows / 640);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back((float)score);
                class_ids.push_back(class_id_point.x);
            }
        }

        // Non-Maximum Suppression (Clean up overlapping boxes)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.7f, 0.45f, indices);

        for (int idx : indices)
        {
            cv::Rect box = boxes[idx];
            int cls = class_ids[idx];
            float conf = confidences[idx];

            // Define Color and Label based on class
            cv::Scalar color;
            std::string label;
            if (cls == 0)
            {
                label = "Closed";
                color = cv::Scalar(0, 0, 255);
            }
            else if (cls == 1)
            {
                label = "Open";
                color = cv::Scalar(0, 255, 0);
            }
            else if (cls == 2)
            {
                label = "Thumb Up";
                color = cv::Scalar(255, 0, 127);
            }
            else if (cls == 3)
            {
                label = "Pray";
                color = cv::Scalar(139, 0, 0);
            }

            // Drawing logic
            int centerX = box.x + box.width / 2;
            int centerY = box.y + box.height / 2;
            int radius = box.width / 3;

            cv::circle(frame, cv::Point(centerX, centerY), radius, color, 2);
            cv::putText(frame, label + " " + std::to_string(conf).substr(0, 4),
                        cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        cv::imshow("C++ Real Time Tracker", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
    return 0;
    
}
