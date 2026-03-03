/**
 * Rohil Kulshreshtha
 * February 14, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Header file for k-means clustering on grayscale intensity values.
 */

#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <opencv2/opencv.hpp>

// Sum of Squared Differences for grayscale values
#define SSD_GREY(a, b) ((int)(a) - (int)(b)) * ((int)(a) - (int)(b))

/**
 * K-means clustering for grayscale intensity values.
 * 
 * @param data Vector of grayscale pixel values (0-255)
 * @param means Vector to store cluster centers
 * @param labels Array to store cluster assignment for each data point
 * @param K Number of clusters
 * @param maxIterations Maximum number of iterations (default: 10)
 * @param stopThresh Stop if mean change is below this threshold (default: 0)
 * @return 0 on success, -1 on error
 */
int kmeans(std::vector<uchar> &data, std::vector<uchar> &means, int *labels, int K, int maxIterations = 10, int stopThresh = 0);

#endif // KMEANS_H