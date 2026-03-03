/**
 * Rohil Kulshreshtha
 * February 16, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Feature computation implementation
 */

#include <opencv2/opencv.hpp>
#include <cmath>
#include "features.h"

int computeCentralMoments(const cv::Mat &labels, int regionLabel, const cv::Point &centroid, double &mu20, double &mu02, double &mu11) {
    mu20 = 0.0;
    mu02 = 0.0;
    mu11 = 0.0;
    
    for (int i = 0; i < labels.rows; i++) {
        const int *labelsPtr = labels.ptr<int>(i);
        for (int j = 0; j < labels.cols; j++) {
            if (labelsPtr[j] == regionLabel) {
                double x = j - centroid.x;
                double y = i - centroid.y;
                
                mu20 += x * x;
                mu02 += y * y;
                mu11 += x * y;
            }
        }
    }
    
    return 0;
}

double computeOrientation(double mu20, double mu02, double mu11) {
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
    return theta;
}

int computeFeatures(const cv::Mat &labels, const RegionStats &stats, RegionFeatures &features) {
    features.label = stats.label;
    features.centroid = cv::Point2d(stats.centroid.x, stats.centroid.y);
    
    double bboxArea = stats.boundingBox.width * stats.boundingBox.height;
    features.percentFilled = (bboxArea > 0) ? (double)stats.area / bboxArea : 0.0;
    
    features.aspectRatio = (stats.boundingBox.height > 0) ? (double)stats.boundingBox.width / stats.boundingBox.height : 0.0;
    
    double mu20, mu02, mu11;
    computeCentralMoments(labels, stats.label, stats.centroid, mu20, mu02, mu11);
    
    features.orientation = computeOrientation(mu20, mu02, mu11);

    computeAxisExtents(labels, stats.label, stats.centroid, features.orientation, features.minE1, features.maxE1, features.minE2, features.maxE2);
    
    double length = std::max(stats.boundingBox.width, stats.boundingBox.height) * 1.2;
    double width = std::min(stats.boundingBox.width, stats.boundingBox.height) * 1.2;
    
    features.orientedBoundingBox = cv::RotatedRect(
        cv::Point2f(features.centroid.x, features.centroid.y),
        cv::Size2f(length, width),
        features.orientation * 180.0 / CV_PI
    );
    
    double m00 = stats.area;
    double nu20 = mu20 / (m00 * m00);
    double nu02 = mu02 / (m00 * m00);
    double nu11 = mu11 / (m00 * m00);
    
    features.huMoments[0] = nu20 + nu02;
    features.huMoments[1] = (nu20 - nu02) * (nu20 - nu02) + 4 * nu11 * nu11;
    
    for (int i = 2; i < 7; i++) {
        features.huMoments[i] = 0.0;
    }
    
    return 0;
}

int computeAllFeatures(const cv::Mat &labels, const std::vector<RegionStats> &stats, std::vector<RegionFeatures> &features) {
    features.clear();
    
    for (const auto &stat : stats) {
        RegionFeatures feat;
        computeFeatures(labels, stat, feat);
        features.push_back(feat);
    }
    
    return 0;
}

int drawFeatures(const cv::Mat &src, const std::vector<RegionFeatures> &features, cv::Mat &dst) {
    src.copyTo(dst);
    
    for (const auto &feat : features) {
        cv::Point center(feat.centroid.x, feat.centroid.y);
        
        int axisLength = 80;
        cv::Point axis_end(
            center.x + axisLength * cos(feat.orientation),
            center.y + axisLength * sin(feat.orientation)
        );
        cv::line(dst, center, axis_end, cv::Scalar(0, 255, 0), 2);
        
        cv::Point2f vertices[4];
        feat.orientedBoundingBox.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(dst, vertices[i], vertices[(i+1)%4], cv::Scalar(255, 0, 0), 2);
        }
        
        cv::circle(dst, center, 5, cv::Scalar(0, 0, 255), -1);
        
        char text[256];
        sprintf(text, "Fill:%.2f AR:%.2f", feat.percentFilled, feat.aspectRatio);
        cv::putText(dst, text, cv::Point(center.x - 80, center.y - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }
    
    return 0;
}

int computeAxisExtents(const cv::Mat &labels, int regionLabel, const cv::Point &centroid, double theta, double &minE1, double &maxE1,  double &minE2, double &maxE2) {
    minE1 = 0.0;
    maxE1 = 0.0;
    minE2 = 0.0;
    maxE2 = 0.0;
    
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);
    
    for (int i = 0; i < labels.rows; i++) {
        const int *labelsPtr = labels.ptr<int>(i);
        for (int j = 0; j < labels.cols; j++) {
            if (labelsPtr[j] == regionLabel) {
                double x = j - centroid.x;
                double y = i - centroid.y;
                
                double e1 = x * cosTheta + y * sinTheta;
                double e2 = -x * sinTheta + y * cosTheta;
                
                if (e1 < minE1) minE1 = e1;
                if (e1 > maxE1) maxE1 = e1;
                if (e2 < minE2) minE2 = e2;
                if (e2 > maxE2) maxE2 = e2;
            }
        }
    }
    
    return 0;
}