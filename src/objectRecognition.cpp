/**
 * Rohil Kulshreshtha
 * February 14, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Main program for 2D object recognition system
 */

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "morphology.h"
#include "segmentation.h"
#include "features.h"
#include "database.h"
#include "classifier.h"

void drawClassificationResults(cv::Mat &img, const std::vector<RegionFeatures> &features, const std::vector<ClassificationResult> &results) {    
    for (size_t i = 0; i < results.size(); i++) {
        if (i >= features.size()) {
            break;
        }
        if (results[i].label.empty()) {
            continue;
        }

        cv::Point pos(features[i].centroid.x - 50, features[i].centroid.y + 40);
        cv::Scalar color = results[i].isUnknown ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        
        char text[256];
        snprintf(text, sizeof(text), "%s (%.2f)", results[i].label.c_str(), results[i].distance);
        
        cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 3);  // Bigger and thicker
    }
}

void processCamera() {
    cv::VideoCapture *capdev;
    
    capdev = new cv::VideoCapture(0);
    // capdev = new cv::VideoCapture("http://<IP-address:port>/video"); // Tried using the IP routing to connect to my phone and use it as a camera (Too slow)

    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        delete capdev;
        return;
    }

    capdev->set(cv::CAP_PROP_BUFFERSIZE, 1);
    // capdev->set(cv::CAP_PROP_FRAME_WIDTH, 640);
    // capdev->set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    int windowWidth, windowHeight;
    if (refS.width > 3000 || refS.height > 3000) {
        windowWidth = refS.width/5;
        windowHeight = refS.height/5;
    } else if (refS.width > 1000 || refS.height > 1000) {
        windowWidth = refS.width/2;
        windowHeight = refS.height/2;
    } else {
        windowWidth = refS.width;
        windowHeight = refS.height;
    }
    

    printf("=== Camera Mode ===\n");
    printf("Press 'q' to quit\n\n");

    // cv::namedWindow("Original", cv::WINDOW_NORMAL);
    // cv::namedWindow("Thresholded", cv::WINDOW_NORMAL);
    // cv::namedWindow("Cleaned", cv::WINDOW_NORMAL);
    // cv::namedWindow("Regions", cv::WINDOW_NORMAL);
    cv::namedWindow("Features", cv::WINDOW_NORMAL);

    // cv::resizeWindow("Original", windowWidth, windowHeight);
    // cv::resizeWindow("Thresholded", windowWidth, windowHeight);
    // cv::resizeWindow("Cleaned", windowWidth, windowHeight);
    // cv::resizeWindow("Regions", windowWidth, windowHeight);
    cv::resizeWindow("Features", windowWidth, windowHeight);


    cv::Mat frame, resized, thresholded, cleaned, labels, regionMap, featuresViz;
    std::vector<RegionStats> stats;
    std::vector<RegionFeatures> features;

    std::vector<DatabaseEntry> database;
    std::vector<double> stdevs;
    std::vector<ClassificationResult> results;
    
    int dbLoaded = loadDatabase("data/object_database.csv", database);
    if (dbLoaded > 0) {
        printf("Loaded %d entries from database\n", dbLoaded);
        computeFeatureStdDevs(database, stdevs);
    } else {
        printf("Warning: Could not load database\n");
    }

    int frameCount = 0;
    int processEveryN = 1;

    for(;;) {
        bool success = capdev->read(frame);
        
        if (!success || frame.empty()) {
            printf("Failed to grab frame\n");
            cv::waitKey(30);
            continue;
        }

        int maxDim = std::max(frame.cols, frame.rows);
        if (maxDim > 800) {
            double scale = 800.0 / maxDim;
            int newWidth = frame.cols * scale;
            int newHeight = frame.rows * scale;
            cv::resize(frame, resized, cv::Size(newWidth, newHeight));
        } else {
            resized = frame;
        }

        frameCount++;
        
        if (frameCount % processEveryN == 0) {
            preprocessAndThreshold(resized, thresholded);
            cleanupBinary(thresholded, cleaned);
            // thresholded.copyTo(cleaned);
            
            segmentRegions(cleaned, labels, stats, 500);
            visualizeRegions(labels, stats, regionMap);
            computeAllFeatures(labels, stats, features);

            results.clear();
            
            int minValidArea = (resized.cols * resized.rows) / 500;

            for (size_t i = 0; i < features.size(); i++) {
                const auto &feat = features[i];
                
                const RegionStats *stat = nullptr;
                for (const auto &s : stats) {
                    if (s.label == feat.label) {
                        stat = &s;
                        break;
                    }
                }
                
                if (stat != nullptr && 
                    stat->area > minValidArea && 
                    !stat->touchesBorder) {
                    
                    ClassificationResult result = classifyObject(feat, database, stdevs, 1, 2.0);
                    results.push_back(result);
                }
            }

            drawFeatures(resized, features, featuresViz);
            drawClassificationResults(featuresViz, features, results);
        }

        // cv::imshow("Original", frame);

        if (!regionMap.empty()) {
            // cv::imshow("Regions", regionMap);
            cv::imshow("Features", featuresViz);
        }
        
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // q or ESC
            break;
        }
    }

    cv::destroyAllWindows();
    delete capdev;
}

void processImage(const char *filename) {
    printf("=== Image Mode ===\n");
    printf("Processing: %s\n\n", filename);

    cv::Mat frame = cv::imread(filename);
    
    if (frame.empty()) {
        printf("Error: Could not read image %s\n", filename);
        return;
    }

    printf("Expected size: %d %d\n", frame.cols, frame.rows);

    int windowWidth, windowHeight;
    if (frame.cols > 3000 || frame.rows > 3000) {
        windowWidth = frame.cols/5;
        windowHeight = frame.rows/5;
    } else if (frame.cols > 1000 || frame.rows > 1000) {
        windowWidth = frame.cols/2;
        windowHeight = frame.rows/2;
    } else {
        windowWidth = frame.cols;
        windowHeight = frame.rows;
    }

    cv::Mat thresholded, resized, cleaned, labels, regionMap, featuresViz;
    std::vector<RegionStats> stats;
    std::vector<RegionFeatures> features;

    int maxDim = (frame.cols > frame.rows) ? frame.cols : frame.rows;
    if (maxDim > 800) {
        double scale = 800.0 / maxDim;
        int newWidth = frame.cols * scale;
        int newHeight = frame.rows * scale;
        cv::resize(frame, resized, cv::Size(newWidth, newHeight));
    } else {
        resized = frame;
    }

    std::vector<DatabaseEntry> database;
    std::vector<double> stdevs;

    loadDatabase("data/object_database.csv", database);
    computeFeatureStdDevs(database, stdevs);
    
    preprocessAndThreshold(resized, thresholded);
    cleanupBinary(thresholded, cleaned);
    
    int numRegions = segmentRegions(cleaned, labels, stats, 100);
    visualizeRegions(labels, stats, regionMap);
    computeAllFeatures(labels, stats, features);

    printf("Found %d regions\n\n", numRegions);
    for (size_t i = 0; i < features.size(); i++) {
        printf("=== Region %d ===\n", features[i].label);
        printf("  Centroid: (%.1f, %.1f)\n", features[i].centroid.x, features[i].centroid.y);
        printf("  Percent Filled: %.3f\n", features[i].percentFilled);
        printf("  Aspect Ratio: %.3f\n", features[i].aspectRatio);
        printf("  Orientation: %.3f rad (%.1f deg)\n", 
               features[i].orientation, features[i].orientation * 180.0 / CV_PI);
        printf("  Hu Moments: [%.6f, %.6f]\n", 
               features[i].huMoments[0], features[i].huMoments[1]);
        printf("\n");
    }

    printf("\n");
    std::vector<ClassificationResult> results;

    int minValidArea = (resized.cols * resized.rows) / 500;

    for (size_t i = 0; i < features.size(); i++) {
        const auto &feat = features[i];
        
        const RegionStats *stat = nullptr;
        for (const auto &s : stats) {
            if (s.label == feat.label) {
                stat = &s;
                break;
            }
        }
        
        if (stat != nullptr && 
            stat->area > minValidArea && 
            !stat->touchesBorder) {
            
            ClassificationResult result = classifyObject(feat, database, stdevs, 1, 2.0);
            results.push_back(result);
            printf("Classified as: %s (distance: %.3f)\n", 
                result.label.c_str(), result.distance);
        } else {
            ClassificationResult dummy;
            dummy.label = "";
            dummy.isUnknown = true;
            results.push_back(dummy);
        }
    }
    printf("\n");

    drawFeatures(resized, features, featuresViz);
    drawClassificationResults(featuresViz, features, results);

    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Thresholded", cv::WINDOW_NORMAL);
    cv::namedWindow("Cleaned", cv::WINDOW_NORMAL);
    cv::namedWindow("Regions", cv::WINDOW_NORMAL);
    cv::namedWindow("Features", cv::WINDOW_NORMAL);

    cv::resizeWindow("Original", windowWidth, windowHeight);
    cv::resizeWindow("Thresholded", windowWidth, windowHeight);
    cv::resizeWindow("Cleaned", windowWidth, windowHeight);
    cv::resizeWindow("Regions", windowWidth, windowHeight);
    cv::resizeWindow("Features", windowWidth, windowHeight);

    cv::imshow("Original", frame);
    cv::imshow("Thresholded", thresholded);
    cv::imshow("Cleaned", cleaned);
    cv::imshow("Regions", regionMap);
    cv::imshow("Features", featuresViz);

    printf("\nPress any key to close\n");
    cv::waitKey(0);

    cv::destroyAllWindows();
}

int main(int argc, char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    if (argc == 1) {
        processCamera();
    } else if (argc == 2) {
        processImage(argv[1]);
    } else {
        printf("Usage:\n");
        printf("  %s              # Camera mode\n", argv[0]);
        printf("  %s <image.jpg>  # Image mode\n", argv[0]);
        return -1;
    }

    return 0;
}