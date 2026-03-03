/**
 * Rohil Kulshreshtha
 * February 14, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Header file for thresholding operations.
 */

#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>

/**
 * Compute dynamic threshold value using k-means clustering (ISODATA algorithm).
 * 
 * @param src Input image
 * @param sampleRate Sample every Nth pixel
 * @return Threshold value (0-255) calculated from k-means
 */
int computeDynamicThreshold(const cv::Mat &src, int sampleRate = 16);

/**
 * Apply custom HSV-based thresholding FROM SCRATCH.
 * 
 * @param src Input BGR image
 * @param dst Output binary image (CV_8UC1): 0=foreground (object), 255=background
 * @param satThreshold Saturation threshold (0-255), default 60
 * @param valThreshold Value/brightness threshold (0-255), default 180
 * @return 0 on success
 */
int hsvThresholdCustom(const cv::Mat &src, cv::Mat &dst, int satThreshold = 60, int valThreshold = 180);

/**
 * Apply adaptive HSV-based thresholding using k-means to compute thresholds.
 * 
 * @param src Input BGR image
 * @param dst Output binary image (CV_8UC1): 0=foreground, 255=background
 * @return 0 on success
 */
int adaptiveHsvThreshold(const cv::Mat &src, cv::Mat &dst);

/**
 * Complete preprocessing and thresholding pipeline.
 * 
 * @param src Input BGR image
 * @param dst Output binary image (CV_8UC1): 0=foreground, 255=background
 * @return 0 on success
 */
int preprocessAndThreshold(const cv::Mat &src, cv::Mat &dst);

/**
 * Complete pipeline: preprocessing, thresholding, and morphological cleanup.
 * 
 * @param src Input BGR image
 * @param dst Output cleaned binary image (CV_8UC1)
 * @return 0 on success
 */
int preprocessThresholdAndCleanup(const cv::Mat &src, cv::Mat &dst);

#endif // THRESHOLD_H