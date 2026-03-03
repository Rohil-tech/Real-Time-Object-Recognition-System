/**
 * Rohil Kulshreshtha
 * February 18, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Classification implementation
 */

#include <cmath>
#include <algorithm>
#include <map>
#include "classifier.h"

int computeFeatureStdDevs(const std::vector<DatabaseEntry> &database, std::vector<double> &stdevs) {
    stdevs.clear();
    stdevs.resize(4, 0.0);
    
    if (database.empty()) {
        return -1;
    }
    
    std::vector<double> means(4, 0.0);
    
    for (const auto &entry : database) {
        means[0] += entry.features.percentFilled;
        means[1] += entry.features.aspectRatio;
        means[2] += entry.features.huMoments[0];
        means[3] += entry.features.huMoments[1];
    }
    
    for (int i = 0; i < 4; i++) {
        means[i] /= database.size();
    }
    
    for (const auto &entry : database) {
        stdevs[0] += (entry.features.percentFilled - means[0]) * (entry.features.percentFilled - means[0]);
        stdevs[1] += (entry.features.aspectRatio - means[1]) * (entry.features.aspectRatio - means[1]);
        stdevs[2] += (entry.features.huMoments[0] - means[2]) * (entry.features.huMoments[0] - means[2]);
        stdevs[3] += (entry.features.huMoments[1] - means[3]) * (entry.features.huMoments[1] - means[3]);
    }
    
    for (int i = 0; i < 4; i++) {
        stdevs[i] = sqrt(stdevs[i] / database.size());
        if (stdevs[i] < 1e-6) {
            stdevs[i] = 1.0;
        }
    }
    
    return 0;
}

double scaledEuclideanDistance(const RegionFeatures &features1, const RegionFeatures &features2, const std::vector<double> &stdevs) {
    double sum = 0.0;
    
    double diff0 = (features1.percentFilled - features2.percentFilled) / stdevs[0];
    double diff1 = (features1.aspectRatio - features2.aspectRatio) / stdevs[1];
    double diff2 = (features1.huMoments[0] - features2.huMoments[0]) / stdevs[2];
    double diff3 = (features1.huMoments[1] - features2.huMoments[1]) / stdevs[3];
    
    sum = diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    
    return sqrt(sum);
}

ClassificationResult classifyObject(const RegionFeatures &features, const std::vector<DatabaseEntry> &database, const std::vector<double> &stdevs, int K, double unknownThreshold) {
    ClassificationResult result;
    result.label = "unknown";
    result.distance = 999999.0;
    result.confidence = 0.0;
    result.isUnknown = true;
    
    if (database.empty()) {
        return result;
    }
    
    std::vector<std::pair<double, std::string>> distances;
    
    for (const auto &entry : database) {
        double dist = scaledEuclideanDistance(features, entry.features, stdevs);
        distances.push_back({dist, entry.label});
    }
    
    std::sort(distances.begin(), distances.end());
    
    if (K == 1) {
        result.distance = distances[0].first;
        result.label = distances[0].second;
        result.isUnknown = (result.distance > unknownThreshold);
        result.confidence = 1.0 / (1.0 + result.distance);
    } else {
        std::map<std::string, int> votes;
        double totalDist = 0.0;
        
        for (int i = 0; i < K && i < distances.size(); i++) {
            votes[distances[i].second]++;
            totalDist += distances[i].first;
        }
        
        int maxVotes = 0;
        for (const auto &vote : votes) {
            if (vote.second > maxVotes) {
                maxVotes = vote.second;
                result.label = vote.first;
            }
        }
        
        result.distance = totalDist / K;
        result.isUnknown = (result.distance > unknownThreshold);
        result.confidence = (double)maxVotes / K;
    }
    
    if (result.isUnknown) {
        result.label = "unknown";
    }
    
    return result;
}