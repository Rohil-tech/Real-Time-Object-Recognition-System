/**
 * Rohil Kulshreshtha
 * February 15, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Morphological filtering implementation
 */

#include <opencv2/opencv.hpp>
#include "morphology.h"

int erode(const cv::Mat &src, cv::Mat &dst, int kernelSize) {
    dst = src.clone();  // Copy entire image first
    
    int offset = kernelSize / 2;
    
    for (int i = offset; i < src.rows - offset; i++) {
        uchar *dstPtr = dst.ptr<uchar>(i);
        
        for (int j = offset; j < src.cols - offset; j++) {
            uchar minVal = 255;
            
            for (int ki = -offset; ki <= offset; ki++) {
                const uchar *kernelRow = src.ptr<uchar>(i + ki);
                for (int kj = -offset; kj <= offset; kj++) {
                    if (kernelRow[j + kj] < minVal) {
                        minVal = kernelRow[j + kj];
                    }
                }
            }
            
            dstPtr[j] = minVal;
        }
    }
    
    return 0;
}

int dilate(const cv::Mat &src, cv::Mat &dst, int kernelSize) {
    dst = src.clone();  // Copy entire image first
    
    int offset = kernelSize / 2;
    
    for (int i = offset; i < src.rows - offset; i++) {
        uchar *dstPtr = dst.ptr<uchar>(i);
        
        for (int j = offset; j < src.cols - offset; j++) {
            uchar maxVal = 0;
            
            for (int ki = -offset; ki <= offset; ki++) {
                const uchar *kernelRow = src.ptr<uchar>(i + ki);
                for (int kj = -offset; kj <= offset; kj++) {
                    if (kernelRow[j + kj] > maxVal) {
                        maxVal = kernelRow[j + kj];
                    }
                }
            }
            
            dstPtr[j] = maxVal;
        }
    }
    
    return 0;
}

int morphOpen(const cv::Mat &src, cv::Mat &dst, int kernelSize) {
    cv::Mat temp;
    erode(src, temp, kernelSize);
    dilate(temp, dst, kernelSize);
    return 0;
}

int morphClose(const cv::Mat &src, cv::Mat &dst, int kernelSize) {
    cv::Mat temp;
    dilate(src, temp, kernelSize);
    erode(temp, dst, kernelSize);
    return 0;
}

int cleanupBinary(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat temp1, temp2;
    
    morphClose(src, temp1, 7);
    morphClose(temp1, temp2, 7);
    morphClose(temp2, temp1, 5);
    morphOpen(temp1, dst, 3);
    
    return 0;
}