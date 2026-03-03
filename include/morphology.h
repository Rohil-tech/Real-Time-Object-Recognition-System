/**
 * Rohil Kulshreshtha
 * February 15, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Morphological filtering operations for binary image cleanup.
 */

#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <opencv2/opencv.hpp>

/**
 * Erosion: Shrinks white regions, removes small white noise.
 * 
 * @param src Input binary image (CV_8UC1)
 * @param dst Output eroded image
 * @param kernelSize Size of erosion kernel (default: 3)
 * @return 0 on success
 */
int erode(const cv::Mat &src, cv::Mat &dst, int kernelSize = 3);

/**
 * Dilation: Grows white regions, fills small black holes.
 * 
 * @param src Input binary image (CV_8UC1)
 * @param dst Output dilated image
 * @param kernelSize Size of dilation kernel (default: 3)
 * @return 0 on success
 */
int dilate(const cv::Mat &src, cv::Mat &dst, int kernelSize = 3);

/**
 * Opening: Erosion followed by dilation.
 * Removes small white noise while preserving object size.
 * 
 * @param src Input binary image (CV_8UC1)
 * @param dst Output opened image
 * @param kernelSize Size of kernel (default: 3)
 * @return 0 on success
 */
int morphOpen(const cv::Mat &src, cv::Mat &dst, int kernelSize = 3);

/**
 * Closing: Dilation followed by erosion.
 * Fills small black holes while preserving object size.
 * 
 * @param src Input binary image (CV_8UC1)
 * @param dst Output closed image
 * @param kernelSize Size of kernel (default: 3)
 * @return 0 on success
 */
int morphClose(const cv::Mat &src, cv::Mat &dst, int kernelSize = 3);

/**
 * Complete cleanup pipeline: opening + closing.
 * 
 * @param src Input binary image (CV_8UC1)
 * @param dst Output cleaned image
 * @return 0 on success
 */
int cleanupBinary(const cv::Mat &src, cv::Mat &dst);

#endif // MORPHOLOGY_H