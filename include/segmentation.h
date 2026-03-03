/**
 * Rohil Kulshreshtha
 * February 15, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Connected components segmentation and region analysis.
 */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

/**
 * Region statistics structure.
 */
struct RegionStats {
    int label;
    int area;
    cv::Point centroid;
    cv::Rect boundingBox;
    bool touchesBorder;
};

/**
 * Two-pass connected components algorithm.
 * 
 * @param src Input binary image (CV_8UC1, 255=foreground, 0=background)
 * @param labels Output label map (each pixel labeled with region ID)
 * @return Number of regions found (including background)
 */
int connectedComponents(const cv::Mat &src, cv::Mat &labels);

/**
 * Compute statistics for each labeled region.
 * 
 * @param labels Label map from connectedComponents
 * @param stats Output vector of region statistics
 * @param minArea Minimum region area to include
 * @return 0 on success
 */
int computeRegionStats(const cv::Mat &labels, std::vector<RegionStats> &stats, int minArea = 100);

/**
 * Segment regions: connected components + filtering.
 * 
 * @param src Input binary image (CV_8UC1)
 * @param labels Output label map
 * @param stats Vector of region statistics for valid regions
 * @param minArea Minimum region area to keep
 * @return Number of valid regions found
 */
int segmentRegions(const cv::Mat &src, cv::Mat &labels, std::vector<RegionStats> &stats, int minArea = 100);

/**
 * Create colored visualization of region map.
 * 
 * @param labels Label map from segmentRegions
 * @param stats Region statistics
 * @param dst Output colored region map (BGR)
 * @return 0 on success
 */
int visualizeRegions(const cv::Mat &labels, const std::vector<RegionStats> &stats, cv::Mat &dst);

/**
 * Get the N largest regions by area.
 * 
 * @param stats Input region statistics
 * @param N Number of largest regions to keep
 * @return Vector of N largest regions
 */
std::vector<RegionStats> getLargestRegions(const std::vector<RegionStats> &stats, int N);

#endif // SEGMENTATION_H