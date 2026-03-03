/**
 * Rohil Kulshreshtha
 * February 14, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Thresholding operations.
 */

#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "filter.h"
#include "kmeans.h"
#include "morphology.h"

int computeDynamicThreshold(const cv::Mat &src, int sampleRate) {
    cv::Mat grey;
    
    if (src.channels() == 3) {
        cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    } else {
        grey = src;
    }

    std::vector<uchar> samples;
    for (int i = 0; i < grey.rows; i += sampleRate) {
        for (int j = 0; j < grey.cols; j += sampleRate) {
            samples.push_back(grey.at<uchar>(i, j));
        }
    }

    if (samples.empty()) {
        return 128;
    }

    std::vector<uchar> means;
    int *labels = new int[samples.size()];

    kmeans(samples, means, labels, 2, 10, 0);

    delete[] labels;

    if (means.size() < 2) {
        return 128;
    }

    int threshold = ((int)means[0] + (int)means[1]) / 2;
    return threshold;
}

int hsvThresholdCustom(const cv::Mat &src, cv::Mat &dst, int satThreshold, int valThreshold) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    dst.create(src.size(), CV_8UC1);

    for (int i = 0; i < hsv.rows; i++) {
        cv::Vec3b *hsvPtr = hsv.ptr<cv::Vec3b>(i);
        uchar *dstPtr = dst.ptr<uchar>(i);
        
        for (int j = 0; j < hsv.cols; j++) {
            uchar S = hsvPtr[j][1];
            uchar V = hsvPtr[j][2];
            
            // // For white background
            // if (V > 210 && S < 30) {
            //     dstPtr[j] = 0;
            // } else if (V < 120 || S > 60) {
            //     dstPtr[j] = 255;  // Foreground
            // } else {
            //     dstPtr[j] = 0;  // Background
            // }

            // Lowered for tan background
            if (V > 150 && S < 50) {
                dstPtr[j] = 0;
            } else if (V < 100 || S > 70) {
                dstPtr[j] = 255;  // Foreground
            } else {
                dstPtr[j] = 0;  // Background
            }
        }
    }

    return 0;
}

int adaptiveHsvThreshold(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // Run k-means on V channel to find brightness threshold
    int valThreshold = computeDynamicThreshold(channels[2], 16);
    
    printf("K-means computed V threshold: %d\n", valThreshold);
    
    // Use k-means result but apply it more intelligently
    return hsvThresholdCustom(src, dst, 80, valThreshold);
}

int preprocessAndThreshold(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat blurred;
    blur5x5_2(const_cast<cv::Mat&>(src), blurred);
    
    // Use fixed thresholds for real-time performance
    hsvThresholdCustom(blurred, dst, 80, 150);
    
    return 0;
}

int preprocessThresholdAndCleanup(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat thresholded;
    
    preprocessAndThreshold(src, thresholded);
    cleanupBinary(thresholded, dst);
    
    return 0;
}