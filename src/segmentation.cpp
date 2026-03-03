/**
 * Rohil Kulshreshtha
 * February 15, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Connected components implementation using two-pass algorithm
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include "segmentation.h"

int connectedComponents(const cv::Mat &src, cv::Mat &labels) {
    labels.create(src.size(), CV_32S);
    labels.setTo(0);
    
    std::map<int, int> equivalence;
    int nextLabel = 1;
    
    // First pass: assign labels and record equivalences
    for (int i = 0; i < src.rows; i++) {
        const uchar *srcPtr = src.ptr<uchar>(i);
        int *labelsPtr = labels.ptr<int>(i);
        
        for (int j = 0; j < src.cols; j++) {
            if (srcPtr[j] == 0) {
                labelsPtr[j] = 0;
                continue;
            }
            
            std::set<int> neighbors;
            
            // Check 4-connected neighbors (left and top)
            if (i > 0 && labels.at<int>(i-1, j) > 0) {
                neighbors.insert(labels.at<int>(i-1, j));
            }
            if (j > 0 && labelsPtr[j-1] > 0) {
                neighbors.insert(labelsPtr[j-1]);
            }
            
            if (neighbors.empty()) {
                labelsPtr[j] = nextLabel;
                equivalence[nextLabel] = nextLabel;
                nextLabel++;
            } else {
                int minLabel = *neighbors.begin();
                labelsPtr[j] = minLabel;
                
                for (int label : neighbors) {
                    if (label != minLabel) {
                        int root1 = label;
                        while (equivalence[root1] != root1) {
                            root1 = equivalence[root1];
                        }
                        int root2 = minLabel;
                        while (equivalence[root2] != root2) {
                            root2 = equivalence[root2];
                        }
                        if (root1 != root2) {
                            equivalence[root1] = root2;
                        }
                    }
                }
            }
        }
    }
    
    // Flatten equivalence table
    for (auto &pair : equivalence) {
        int root = pair.first;
        while (equivalence[root] != root) {
            root = equivalence[root];
        }
        pair.second = root;
    }
    
    // Relabel to sequential IDs
    std::map<int, int> newLabels;
    int newLabel = 1;
    for (auto &pair : equivalence) {
        int root = pair.second;
        if (newLabels.find(root) == newLabels.end()) {
            newLabels[root] = newLabel++;
        }
        pair.second = newLabels[root];
    }
    
    // Second pass: replace labels with equivalence roots
    for (int i = 0; i < labels.rows; i++) {
        int *labelsPtr = labels.ptr<int>(i);
        for (int j = 0; j < labels.cols; j++) {
            if (labelsPtr[j] > 0) {
                labelsPtr[j] = equivalence[labelsPtr[j]];
            }
        }
    }
    
    return newLabel;
}

int computeRegionStats(const cv::Mat &labels, std::vector<RegionStats> &stats, int minArea) {
    stats.clear();
    
    std::map<int, RegionStats> regionMap;
    std::map<int, long long> sumX;
    std::map<int, long long> sumY;
    
    for (int i = 0; i < labels.rows; i++) {
        const int *labelsPtr = labels.ptr<int>(i);
        for (int j = 0; j < labels.cols; j++) {
            int label = labelsPtr[j];
            if (label == 0) continue;
            
            if (regionMap.find(label) == regionMap.end()) {
                RegionStats rs;
                rs.label = label;
                rs.area = 0;
                rs.centroid = cv::Point(0, 0);
                rs.boundingBox = cv::Rect(j, i, 0, 0);
                rs.touchesBorder = false;
                regionMap[label] = rs;
                sumX[label] = 0;
                sumY[label] = 0;
            }
            
            RegionStats &rs = regionMap[label];
            rs.area++;
            sumX[label] += j;
            sumY[label] += i;
            
            if (rs.boundingBox.width == 0) {
                rs.boundingBox = cv::Rect(j, i, 1, 1);
            } else {
                int x1 = std::min(rs.boundingBox.x, j);
                int y1 = std::min(rs.boundingBox.y, i);
                int x2 = std::max(rs.boundingBox.x + rs.boundingBox.width, j + 1);
                int y2 = std::max(rs.boundingBox.y + rs.boundingBox.height, i + 1);
                rs.boundingBox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            }
            
            if (i == 0 || i == labels.rows - 1 || j == 0 || j == labels.cols - 1) {
                rs.touchesBorder = true;
            }
        }
    }
    
    for (auto &pair : regionMap) {
        RegionStats &rs = pair.second;
        if (rs.area > 0) {
            rs.centroid.x = sumX[rs.label] / rs.area;
            rs.centroid.y = sumY[rs.label] / rs.area;
        }
        
        if (rs.area >= minArea) {
            stats.push_back(rs);
        }
    }
    
    return 0;
}

int segmentRegions(const cv::Mat &src, cv::Mat &labels, std::vector<RegionStats> &stats, int minArea) {
    connectedComponents(src, labels);
    computeRegionStats(labels, stats, minArea);
    return stats.size();
}

int visualizeRegions(const cv::Mat &labels, const std::vector<RegionStats> &stats, cv::Mat &dst) {
    dst.create(labels.size(), CV_8UC3);
    dst.setTo(cv::Scalar(0, 0, 0));
    
    std::map<int, cv::Vec3b> colors;
    for (const auto &rs : stats) {
        cv::Vec3b color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);
        colors[rs.label] = color;
    }
    
    for (int i = 0; i < labels.rows; i++) {
        const int *labelsPtr = labels.ptr<int>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        for (int j = 0; j < labels.cols; j++) {
            int label = labelsPtr[j];
            if (label > 0 && colors.find(label) != colors.end()) {
                dstPtr[j] = colors[label];
            }
        }
    }
    
    return 0;
}

std::vector<RegionStats> getLargestRegions(const std::vector<RegionStats> &stats, int N) {
    std::vector<RegionStats> sorted = stats;
    std::sort(sorted.begin(), sorted.end(), 
            [](const RegionStats &a, const RegionStats &b) {
                return a.area > b.area;
            });
    
    if (sorted.size() > N) {
        sorted.resize(N);
    }
    
    return sorted;
}