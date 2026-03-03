/**
 * Rohil Kulshreshtha
 * February 16, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Feature computation for region analysis.
 */

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "segmentation.h"

/**
 * Feature vector for a region.
 */
struct RegionFeatures {
    int label;
    double percentFilled;
    double aspectRatio;
    double orientation;
    cv::Point2d centroid;
    cv::RotatedRect orientedBoundingBox;
    double huMoments[7];
    double minE1, maxE1, minE2, maxE2;
};

/**
 * Compute central moments for a region.
 * 
 * @param labels Label map
 * @param regionLabel Which region to analyze
 * @param centroid Region centroid
 * @param mu20 Output: central moment μ20
 * @param mu02 Output: central moment μ02
 * @param mu11 Output: central moment μ11
 * @return 0 on success
 */
int computeCentralMoments(const cv::Mat &labels, int regionLabel, const cv::Point &centroid, double &mu20, double &mu02, double &mu11);

/**
 * Compute axis of least central moment (orientation).
 * 
 * @param mu20 Central moment μ20
 * @param mu02 Central moment μ02
 * @param mu11 Central moment μ11
 * @return Orientation angle in radians
 */
double computeOrientation(double mu20, double mu02, double mu11);

/**
 * Compute all features for a region.
 * 
 * @param labels Label map
 * @param stats Region statistics
 * @param features Output feature vector
 * @return 0 on success
 */
int computeFeatures(const cv::Mat &labels, const RegionStats &stats, RegionFeatures &features);

/**
 * Compute features for all regions.
 * 
 * @param labels Label map
 * @param stats Vector of region statistics
 * @param features Output vector of feature vectors
 * @return 0 on success
 */
int computeAllFeatures(const cv::Mat &labels, const std::vector<RegionStats> &stats, std::vector<RegionFeatures> &features);

/**
 * Draw features on image (axis, oriented bbox, text).
 * 
 * @param src Input image (BGR)
 * @param features Region features
 * @param dst Output image with overlays
 * @return 0 on success
 */
int drawFeatures(const cv::Mat &src, const std::vector<RegionFeatures> &features, cv::Mat &dst);

/**
 * Compute extent of region along primary and secondary axes.
 * Projects all region pixels onto the principal axes to find min/max extents.
 * 
 * @param labels Label map
 * @param regionLabel Region to analyze
 * @param centroid Region centroid
 * @param theta Orientation angle (radians)
 * @param minE1 Output: minimum extent along primary axis (negative)
 * @param maxE1 Output: maximum extent along primary axis (positive)
 * @param minE2 Output: minimum extent along secondary axis (negative)
 * @param maxE2 Output: maximum extent along secondary axis (positive)
 * @return 0 on success
 */
int computeAxisExtents(const cv::Mat &labels, int regionLabel, const cv::Point &centroid, double theta, double &minE1, double &maxE1,  double &minE2, double &maxE2);

#endif // FEATURES_H