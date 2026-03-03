/**
 * Rohil Kulshreshtha
 * February 11, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Header file for image filtering operations.
 */

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

/**
 * Apply 5x5 Gaussian blur using separable filters.
 * 
 * @param src Input image (BGR or grayscale)
 * @param dst Output blurred image
 * @return 0 on success
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

#endif // FILTER_H