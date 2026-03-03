/**
 * Rohil Kulshreshtha
 * February 18, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Classification system for object recognition.
 */

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include "database.h"

/**
 * Classification result.
 */
struct ClassificationResult {
    std::string label;
    float distance;
    float confidence;
    bool isUnknown;
};

/**
 * Compute standard deviations for each feature dimension.
 * 
 * @param database Vector of database entries
 * @param stdevs Output vector of standard deviations [4 values]
 * @return 0 on success
 */
int computeFeatureStdDevs(const std::vector<DatabaseEntry> &database, std::vector<double> &stdevs);

/**
 * Compute scaled Euclidean distance between two feature vectors.
 * 
 * @param features1 First feature vector
 * @param features2 Second feature vector
 * @param stdevs Standard deviations for scaling
 * @return Scaled Euclidean distance
 */
double scaledEuclideanDistance(const RegionFeatures &features1, const RegionFeatures &features2, const std::vector<double> &stdevs);

/**
 * Classify object using K-nearest neighbors.
 * 
 * @param features Feature vector of unknown object
 * @param database Training database
 * @param stdevs Feature standard deviations
 * @param K Number of neighbors (default: 1)
 * @param unknownThreshold Distance threshold for unknown detection
 * @return Classification result
 */
ClassificationResult classifyObject(const RegionFeatures &features, const std::vector<DatabaseEntry> &database, const std::vector<double> &stdevs, int K = 1, double unknownThreshold = 2.0);

#endif // CLASSIFIER_H