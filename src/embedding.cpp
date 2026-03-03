/**
 * Rohil Kulshreshtha
 * February 19, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * CNN embedding extraction using ResNet18 via OpenCV DNN
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include "embedding.h"

int loadResNet18(const char *modelPath, cv::dnn::Net &net) {
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (net.empty()) {
            printf("Error: Could not load model from %s\n", modelPath);
            return -1;
        }
        printf("ResNet18 model loaded successfully\n");
        return 0;
    } catch (const cv::Exception &e) {
        printf("Error loading model: %s\n", e.what());
        return -1;
    }
}

int prepareROI(const cv::Mat &frame, const RegionFeatures &features, cv::Mat &roiImage, int targetSize) {
    
    int cx = features.centroid.x;
    int cy = features.centroid.y;
    double theta = features.orientation;
    
    cv::Mat rotatedImage;
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), -theta * 180.0 / CV_PI, 1.0);
    
    int maxDim = (frame.cols > frame.rows) ? frame.cols : frame.rows;
    int largest = (int)(1.414 * maxDim);
    cv::warpAffine(frame, rotatedImage, M, cv::Size(largest, largest));
    
    int left = cx + (int)features.minE1;
    int top = cy - (int)features.maxE2;
    int width = (int)(features.maxE1 - features.minE1);
    int height = (int)(features.maxE2 - features.minE2);
    
    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    if (left + width >= rotatedImage.cols) width = rotatedImage.cols - 1 - left;
    if (top + height >= rotatedImage.rows) height = rotatedImage.rows - 1 - top;
    
    if (width <= 0 || height <= 0) {
        printf("Error: Invalid ROI dimensions\n");
        return -1;
    }
    
    cv::Rect objROI(left, top, width, height);
    cv::Mat extracted = rotatedImage(objROI);
    
    cv::resize(extracted, roiImage, cv::Size(targetSize, targetSize));
    
    return 0;
}

int getEmbedding(const cv::Mat &roiImage, cv::Mat &embedding, cv::dnn::Net &net) {
    const int networkSize = 224;
    cv::Mat blob;
    
    cv::dnn::blobFromImage(roiImage, blob, 1.0/255.0, cv::Size(networkSize, networkSize), cv::Scalar(0, 0, 0), true, false, CV_32F);
    
    net.setInput(blob);
    embedding = net.forward();
    embedding = embedding.reshape(1, 1);
    
    return 0;
}

double embeddingDistance(const cv::Mat &emb1, const cv::Mat &emb2) {
    if (emb1.total() != emb2.total()) {
        return -1.0;
    }
    
    double sum = 0.0;
    for (int i = 0; i < emb1.total(); i++) {
        double diff = emb1.at<float>(i) - emb2.at<float>(i);
        sum += diff * diff;
    }
    
    return sum;
}