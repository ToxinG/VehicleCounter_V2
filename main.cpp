#include "Blob.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <unistd.h>

const cv::Scalar BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar BLUE = cv::Scalar(255.0, 0.0, 0.0);

const std::string TXT_EXT = ".txt";
const std::string XML_EXT = ".xml";
const char ESC_KEY = 27;
const char SPACE_KEY = 32;
const std::string extensions[] = {"", TXT_EXT, XML_EXT};
const int EXT_NUMBER = 3;



void readAndLogVideo(cv::VideoCapture &videoCapture, int extensionCode, std::string logName);
void playVideoWithMarkup(cv::VideoCapture &videoCapture, std::ifstream &log, int extensionCode);
void track2Frames(cv::Mat &prevFrame, cv::Mat &curFrame, std::vector<Blob> &blobs);
void log2FramesTXT(std::vector<Blob> &blobs, std::ofstream &outputFile);
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void updateExistingBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &index);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(const cv::Point& point1, const cv::Point& point2);


int main() {
    int action;
    int logType;
    std::string path;
    std::string name;
    cv::VideoCapture videoCapture;
    while (true) {
        std::cout << "Type \"1\" to read the video and log tracked objects." << std::endl
                  << "Type \"2\" to play the video with markup for tracked objects." << std::endl
                  << "Type \"0\" to exit." << std::endl;
        std::cin >> action;

        //Read and log
        switch (action) {
            case 1:
                //Choose video
                while (true) {
                    std::cout << "Enter path to the video you want to log." << std::endl;
                    std::cin >> path;
                    if (path == "0") {
                        return 0;
                    }
                    videoCapture.open(path);
                    if (!videoCapture.isOpened()) {
                        std::cout << "Cannot open the video. Check the path, please." << std::endl;
                        continue;
                    }

                    //for debugging
                    //std::cout << videoCapture.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;

                    if (videoCapture.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
                        std::cout << "Cannot track anything on a \"video\" with least than two frames." << std::endl
                                  << "Choose another video, please." << std::endl;
                        continue;
                    }
                    break;
                }
                //Choose type for log
                std::cout << "Choose type for the file your video will be logged to." << std:: endl
                          << "Type \"1\" for .txt." << std::endl;
                std::cin >> logType;
                if (path.find_last_of('/') == std::string::npos) {
                    name = path.substr(0, path.find_last_of('.'));
                } else {
                    name = path.substr(path.find_last_of('/') + 1);
                    name = name.substr(0, name.find_last_of('.'));
                }

                readAndLogVideo(videoCapture, logType, name);
                break;

            //Play video with markup
            case 2:
                std::cout << "Enter path to the video you want to play." << std::endl;
                std::cin >> path;
                if (path == "0") {
                    return 0;
                }
                if (path.find_last_of('/') == std::string::npos) {
                    name = path.substr(0, path.find_last_of('.'));
                } else {
                    name = path.substr(path.find_last_of('/') + 1);
                    name = name.substr(0, name.find_last_of('.'));
                }
                logType = 0;
                for (int i = 1; i < EXT_NUMBER; i++) {
                    std::ifstream log ("tracking_logs/" + name + extensions[i]);
                    if (log.is_open()) {
                        logType = i;
                        videoCapture.open(path);
                        playVideoWithMarkup(videoCapture, log, i);
                        break;
                    }
                }
                if (logType == 0) {
                    std::cout << "To play this video with markup it have to be logged first." << std::endl;
                }
                break;

            case 0:
                return 0;

            default:
                break;
        }

    }
}

void readAndLogVideo(cv::VideoCapture &videoCapture, int extensionCode, std::string logName) {
    cv::Mat prevFrame, curFrame;
    std::vector<Blob> blobs;
    int frameNumber = 1;
    mkdir("tracking_logs", S_IRWXU);
    std::ofstream log ("tracking_logs/" + logName + extensions[extensionCode]);
    videoCapture.read(prevFrame);
    videoCapture.read(curFrame);
    while (videoCapture.isOpened()) {
        try {
            track2Frames(prevFrame, curFrame, blobs);
        } catch(cv::Exception &e) {
            std::cout << e.msg;
            std::cout << "That's all Folks!" << std::endl;
            break;
        }
        switch (extensionCode) {
            case 1:
                log << frameNumber++ << " ";
                log2FramesTXT(blobs, log);
                break;
        }

        prevFrame = curFrame.clone();
        if ((videoCapture.get(CV_CAP_PROP_POS_FRAMES) + 1) < videoCapture.get(CV_CAP_PROP_FRAME_COUNT)) {
            videoCapture.read(curFrame);
        }
        else {
            std::cout << "end of video\n";
            break;
        }

        //for debugging
//        std::cout << frameNumber << " " << blobs.size() << std::endl;
//        cv::imshow("curFrame", curFrame);

    }
    log.close();
}

void track2Frames(cv::Mat &prevFrame, cv::Mat &curFrame, std::vector<Blob> &blobs) {
    std::vector<Blob> curFrameBlobs;
    cv::Mat prevFrameCopy = prevFrame.clone();
    cv::Mat curFrameCopy = curFrame.clone();
    cv::Mat imgDifference;
    cv::Mat imgThreshold;
    cv::cvtColor(prevFrameCopy, prevFrameCopy, CV_BGR2GRAY);
    cv::cvtColor(curFrameCopy, curFrameCopy, CV_BGR2GRAY);
    cv::GaussianBlur(prevFrameCopy, prevFrameCopy, cv::Size(5, 5), 0);
    cv::GaussianBlur(curFrameCopy, curFrameCopy, cv::Size(5, 5), 0);
    cv::absdiff(prevFrameCopy, curFrameCopy, imgDifference);
    cv::threshold(imgDifference, imgThreshold, 30, 255.0, CV_THRESH_BINARY);
    cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    for (unsigned int i = 0; i < 2; i++) {
        cv::dilate(imgThreshold, imgThreshold, structuringElement5x5);
        cv::dilate(imgThreshold, imgThreshold, structuringElement5x5);
        cv::erode(imgThreshold, imgThreshold, structuringElement5x5);
    }

    cv::Mat imgThresholdCopy = imgThreshold.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imgThresholdCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //for debugging
//    std::cout << contours.size() << std::endl;

    std::vector<std::vector<cv::Point>> convexHulls(contours.size());

    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::convexHull(contours[i], convexHulls[i]);
    }

    //for debugging
//    std::cout << convexHulls.size() << std::endl;

    for (auto &convexHull : convexHulls) {
        Blob possibleBlob(convexHull);

        if (possibleBlob.currentBoundingRect.area() > 32000 &&
            possibleBlob.dblCurrentAspectRatio > 1.2 &&
            possibleBlob.dblCurrentAspectRatio < 4.0 &&
            possibleBlob.currentBoundingRect.width > 128 &&
            possibleBlob.currentBoundingRect.height > 128 &&
            possibleBlob.dblCurrentDiagonalSize > 256.0 &&
            (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
            curFrameBlobs.push_back(possibleBlob);
        }
    }

    //for debugging
//    std::cout << curFrameBlobs.size() << std::endl;
//    cv::imshow("curFrameCopy", curFrameCopy);

    matchCurrentFrameBlobsToExistingBlobs(blobs, curFrameBlobs);


}

void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {
    for (auto &existingBlob : existingBlobs) {
        existingBlob.blnCurrentMatchFoundOrNewBlob = false;
        existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {
        int indexOfLeastDistance = 0;
        double leastDistance = 100000.0;

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked) {
                double distance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (distance < leastDistance) {
                    leastDistance = distance;
                    indexOfLeastDistance = i;
                }
            }
        }

        if (leastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            updateExistingBlob(currentFrameBlob, existingBlobs, indexOfLeastDistance);
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }

    }

    for (auto &existingBlob : existingBlobs) {
        if (existingBlob.blnCurrentMatchFoundOrNewBlob) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }
        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;
        }
    }
}

void updateExistingBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &index) {
    existingBlobs[index].currentContour = currentFrameBlob.currentContour;
    existingBlobs[index].currentBoundingRect = currentFrameBlob.currentBoundingRect;
    existingBlobs[index].centerPositions.push_back(currentFrameBlob.centerPositions.back());
    existingBlobs[index].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[index].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
    existingBlobs[index].blnStillBeingTracked = true;
    existingBlobs[index].blnCurrentMatchFoundOrNewBlob = true;
}


void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
    existingBlobs.push_back(currentFrameBlob);
}


double distanceBetweenPoints(const cv::Point& point1, const cv::Point& point2) {
    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

void log2FramesTXT(std::vector<Blob> &blobs, std::ofstream &outputFile) {
    for (unsigned int i = 0; i < blobs.size(); i++) {
        if (blobs[i].blnStillBeingTracked) {
            outputFile << i << " " << blobs[i].currentBoundingRect.x << " " << blobs[i].currentBoundingRect.y
                       << " " << blobs[i].currentBoundingRect.width << " " << blobs[i].currentBoundingRect.height << " ";
        }
    }
    outputFile << std::endl;
}

void playVideoWithMarkup(cv::VideoCapture &videoCapture, std::ifstream &log, int extensionCode) {
    char checkForKey = 0;
    bool paused = false;
    cv::Mat frame;
    videoCapture.read(frame);
    double fontScale = (frame.rows * frame.cols) / 300000.0;
    int fontThickness = (int)std::round(fontScale * 1.0);
    switch (extensionCode) {
        case 1:
            std::string s;
            while (getline(log, s) && checkForKey != ESC_KEY) {
                videoCapture.read(frame);
                std::istringstream ss(s);
                int frameNumber, blob, x, y, width, height;
                ss >> frameNumber;
                while (ss >> blob) {
                    ss >> x >> y >> width >> height;
                    cv::rectangle(frame, cv::Point(x, y), cv::Point(x + width, y + height), RED, 2);
                    cv::putText(frame, std::to_string(blob), cv::Point(x + width / 2, y + height / 2),
                            CV_FONT_HERSHEY_SIMPLEX, fontScale, GREEN, fontThickness);
                }
                cv::imshow("VideoWithMarkup", frame);

                if (!paused) {
                    checkForKey = cv::waitKey(15);
                    paused = (checkForKey == SPACE_KEY);
                }

                if (paused) {
                    checkForKey = cv::waitKey(0);
                    if (checkForKey == SPACE_KEY)
                        paused = false;
                }
            }
            if (checkForKey != ESC_KEY) {
                std::cout << "end of video\n";
                cv::waitKey(0);
            }
            return;
    }
}