/**
 * Rohil Kulshreshtha
 * February 14, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Modified K-means implementation for grayscale intensity values
 */

#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "kmeans.h"

int kmeans(std::vector<uchar> &data, std::vector<uchar> &means, int *labels, int K, int maxIterations, int stopThresh) {

    if (K > data.size()) {
        printf("Error: K must be less than number of data points\n");
        return -1;
    }

    means.clear();

    // Initialize K means using comb sampling
    int delta = data.size() / K;
    int istep = rand() % (data.size() % K);
    for (int i = 0; i < K; i++) {
        int index = (istep + i * delta) % data.size();
        means.push_back(data[index]);
    }

    // E-M iterations
    for (int i = 0; i < maxIterations; i++) {

        // Assign each data point to nearest mean
        for (int j = 0; j < data.size(); j++) {
            int minssd = SSD_GREY(means[0], data[j]);
            int minidx = 0;
            for (int k = 1; k < K; k++) {
                int tssd = SSD_GREY(means[k], data[j]);
                if (tssd < minssd) {
                    minssd = tssd;
                    minidx = k;
                }
            }
            labels[j] = minidx;
        }

        // Calculate new means
        std::vector<int> sums(K, 0);
        std::vector<int> counts(K, 0);
        
        for (int j = 0; j < data.size(); j++) {
            sums[labels[j]] += data[j];
            counts[labels[j]]++;
        }

        int totalChange = 0;
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                uchar newMean = sums[k] / counts[k];
                totalChange += SSD_GREY(newMean, means[k]);
                means[k] = newMean;
            }
        }

        if (totalChange <= stopThresh) {
            break;
        }
    }

    return 0;
}