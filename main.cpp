#include "Blob.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <pqxx/pqxx>

const cv::Scalar BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar BLUE = cv::Scalar(255.0, 0.0, 0.0);

const std::string TXT_EXT = ".txt";
const std::string XML_EXT = ".xml";
const std::string DB = "database";
const char ESC_KEY = 27;
const char SPACE_KEY = 32;
const std::string logTypes[] = {"", TXT_EXT, DB};
const int TYPES_NUMBER = (sizeof(logTypes)/sizeof(*logTypes)) - 1;


void readVideoLogToFile(cv::VideoCapture &videoCapture, int logTypeCode, const std::string& logName);
void readVideoLogToDB(cv::VideoCapture &videoCapture, const std::string& tableName);
void playVideoWithMarkupFromFile(cv::VideoCapture &videoCapture, std::ifstream &log, int logTypeCode);
void playVideoWithMarkupFromDB(cv::VideoCapture &videoCapture, std::string &name);
void track2Frames(cv::Mat &prevFrame, cv::Mat &curFrame, std::vector<Blob> &blobs);
void log2FramesTXT(std::vector<Blob> &blobs, std::ofstream &outputFile);
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void updateExistingBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &index);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(const cv::Point& point1, const cv::Point& point2);
std::string loggedTableName(std::ifstream &f, std::string s);


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
        std::getchar();

        //Read and log
        switch (action) {
            case 1:
                //Choose video
                while (true) {
                    std::cout << "Enter path to the video you want to log." << std::endl;
                    std::getline(std::cin, path);

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
                        std::cout << "Cannot track anything on a \"video\" with less than two frames." << std::endl
                                  << "Choose another video, please." << std::endl;
                        continue;
                    }
                    break;
                }
                //Choose type for log
                std::cout << "Choose which format your video will be logged in." << std:: endl;
                for (int i = 1; i <= TYPES_NUMBER; i++) {
                    std::cout << "Type " + std::to_string(i) + " for " + logTypes[i] << std::endl;
                }
                std::cin >> logType;
                if (path.find_last_of('/') == std::string::npos) {
                    name = path.substr(0, path.find_last_of('.'));
                } else {
                    path = path.substr(path.find_last_of('/') + 1);
                    name = path.substr(0, path.find_last_of('.'));
                }
                if (logTypes[logType] == DB) {
                    std::string tableName = name;
                    std::replace(tableName.begin(), tableName.end(), ' ', '_');
                    tableName = "TABLE_" + tableName;
                    readVideoLogToDB(videoCapture, tableName);

                    std::ifstream dbtables("dbtables.txt");
                    if (dbtables.is_open()) {
                        std::string lname = loggedTableName(dbtables, path);
                        if (lname.empty()) {
                            dbtables.close();
                            std::ofstream dbtables("dbtables.txt", std::fstream::app);
                            dbtables << path << " " << tableName << std::endl;
                        }
                    }
                    dbtables.close();
                } else {
                    readVideoLogToFile(videoCapture, logType, name);
                }
                break;

            //Play video with markup
            case 2:
                std::cout << "Enter path to the video you want to play." << std::endl;
                std::cin >> path;
                if (path == "0") {
                    return 0;
                }
                videoCapture.open(path);
                if (path.find_last_of('/') == std::string::npos) {
                    name = path.substr(0, path.find_last_of('.'));
                } else {
                    path = path.substr(path.find_last_of('/') + 1);
                    name = path.substr(0, path.find_last_of('.'));
                }
                logType = 0;
                for (int i = 1; i < TYPES_NUMBER; i++) {
                    std::ifstream log ("tracking_logs/" + name + logTypes[i]);
                    if (log.is_open()) {
                        logType = i;
                        playVideoWithMarkupFromFile(videoCapture, log, i);
                        break;
                    }
                }
                if (logType == 0) {
                    std::ifstream dbtables("dbtables.txt");
                    if (dbtables.is_open()) {
                        std::string lname = loggedTableName(dbtables, path);
                        if (!lname.empty()) {
                            playVideoWithMarkupFromDB(videoCapture, lname);
                            logType = TYPES_NUMBER;
                        }
                    }
                }

                if (logType == 0) {
                    std::cout << "No logs found for this video. To play it with markup it have to be logged first." << std::endl;
                }
                break;

            case 0:
                return 0;

            default:
                break;
        }

    }
}


void readVideoLogToFile(cv::VideoCapture &videoCapture, int logTypeCode, const std::string& logName) {
    cv::Mat prevFrame, curFrame;
    std::vector<Blob> blobs;
    std::ofstream log;
    mkdir("tracking_logs", S_IRWXU);
    log = std::ofstream("tracking_logs/" + logName + logTypes[logTypeCode]);

    int frameNumber = 1;
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
        switch (logTypeCode) {
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


void readVideoLogToDB(cv::VideoCapture &videoCapture, const std::string& tableName) {
    cv::Mat prevFrame, curFrame;
    std::vector<Blob> blobs;

    try {
        pqxx::connection C(
                "dbname = vehicle_counter_db user = vehicle_counter password = vc12345 hostaddr=127.0.0.1 port=5432");
        std::cout << "Connected to " << C.dbname() << std::endl;
        pqxx::work W(C);

        W.exec("DROP TABLE " + tableName + ";");
        W.exec("CREATE TABLE " + tableName + "(" \
        "FRAME_ID INT NOT NULL, " \
        "BLOB_ID INT NOT NULL, " \
        "X INT NOT NULL, " \
        "Y INT NOT NULL, " \
        "WIDTH INT NOT NULL, " \
        "HEIGHT INT NOT NULL, " \
        "CONSTRAINT " + tableName + "_PK PRIMARY KEY (FRAME_ID, BLOB_ID));");

        int frameNumber = 1;
        videoCapture.read(prevFrame);
        videoCapture.read(curFrame);

        while (videoCapture.isOpened()) {
            try {
                track2Frames(prevFrame, curFrame, blobs);
            } catch (cv::Exception &e) {
                std::cout << e.msg;
                std::cout << "That's all Folks!" << std::endl;
                break;
            }

            std::string insert;
            for (unsigned int i = 0; i < blobs.size(); i++) {
                if (blobs[i].blnStillBeingTracked) {
                    insert = "INSERT INTO " + tableName + " (FRAME_ID, BLOB_ID, X, Y, WIDTH, HEIGHT) " +
                             "VALUES (" + std::to_string(frameNumber) + ", " +
                             std::to_string(i) + ", " +
                             std::to_string(blobs[i].currentBoundingRect.x) + ", " +
                             std::to_string(blobs[i].currentBoundingRect.y) + ", " +
                             std::to_string(blobs[i].currentBoundingRect.width) + ", " +
                             std::to_string(blobs[i].currentBoundingRect.height) + ");";
                    W.exec(insert);
                }
            }

            frameNumber++;
            prevFrame = curFrame.clone();
            if ((videoCapture.get(CV_CAP_PROP_POS_FRAMES) + 1) < videoCapture.get(CV_CAP_PROP_FRAME_COUNT)) {
                videoCapture.read(curFrame);
            } else {
                std::cout << "end of video\n";
                break;
            }
        }
        W.commit();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
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
            outputFile << i << " " << blobs[i].currentBoundingRect.x
                       << " " << blobs[i].currentBoundingRect.y
                       << " " << blobs[i].currentBoundingRect.width
                       << " " << blobs[i].currentBoundingRect.height << " ";
        }
    }
    outputFile << std::endl;
}


//TODO: transform switch, make player class

void playVideoWithMarkupFromFile(cv::VideoCapture &videoCapture, std::ifstream &log, int logTypeCode) {
    char checkForKey = 0;
    bool paused = false;
    cv::Mat frame;
    videoCapture.read(frame);
    double fontScale = (frame.rows * frame.cols) / 300000.0;
    int fontThickness = (int)std::round(fontScale * 1.0);
    switch (logTypeCode) {
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
            cv::destroyAllWindows();
            return;
    }
}


void playVideoWithMarkupFromDB(cv::VideoCapture &videoCapture, std::string &name) {
    char checkForKey = 0;
    bool paused = false;
    int frameCount = 1;
    cv::Mat frame;
    videoCapture.read(frame);
    double fontScale = (frame.rows * frame.cols) / 300000.0;
    int fontThickness = (int)std::round(fontScale * 1.0);

    try {
        pqxx::connection C(
                "dbname = vehicle_counter_db user = vehicle_counter password = vc12345 hostaddr=127.0.0.1 port=5432");
        std::cout << "Connected to " << C.dbname() << std::endl;
        pqxx::work W(C);
        pqxx::result R = W.exec("SELECT count(*) FROM " + name + ";");
        int blob, x, y, width, height;
        int count = std::stoi(R[0][0].c_str()); // is there a less clumsy way?
        while (count > 0 && checkForKey != ESC_KEY) {
            R = W.exec("SELECT * FROM " + name + " WHERE FRAME_ID = " + std::to_string(frameCount) + ";");
            videoCapture.read(frame);
            for (const auto& row : R) {
                blob = std::stoi(row[1].c_str());
                x = std::stoi(row[2].c_str());
                y = std::stoi(row[3].c_str());
                width = std::stoi(row[4].c_str());
                height = std::stoi(row[5].c_str());
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

            frameCount++;
            count -= R.size();
        }
        if (checkForKey != ESC_KEY) {
            std::cout << "end of video\n";
            cv::waitKey(0);
        }
        cv::destroyAllWindows();
        return;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}


std::string loggedTableName(std::ifstream &f, std::string sp) {
    std::string s, s1, s2;
    while (getline(f, s)) {
        std::istringstream ss(s);
        ss >> s1 >> s2;
        if (s1 == sp) {
            return s2;
        }
    }
    return "";
}
//         pqxx::connection C("dbname = vehicle_counter_db user = vehicle_counter password = vc12345 hostaddr=127.0.0.1 port=5432");