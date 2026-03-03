/**
 * Rohil Kulshreshtha
 * February 19, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Evaluation program to test classifier accuracy and generate confusion matrix.
 */

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include "threshold.h"
#include "morphology.h"
#include "segmentation.h"
#include "features.h"
#include "database.h"
#include "classifier.h"

int main(int argc, char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    if (argc < 3) {
        printf("Usage: %s <test_directory> <database_csv>\n", argv[0]);
        printf("Example: %s data/test_images data/object_database.csv\n", argv[0]);
        return -1;
    }
    
    const char *testDir = argv[1];
    const char *databaseFile = argv[2];
    
    printf("=== Evaluating Object Recognition System ===\n");
    printf("Test directory: %s\n", testDir);
    printf("Database: %s\n\n", databaseFile);
    
    std::vector<DatabaseEntry> database;
    std::vector<double> stdevs;
    
    int dbLoaded = loadDatabase(databaseFile, database);
    if (dbLoaded <= 0) {
        printf("Error: Could not load database\n");
        return -1;
    }
    
    printf("Loaded %d entries from database\n", dbLoaded);
    computeFeatureStdDevs(database, stdevs);
    printf("Computed feature standard deviations\n\n");
    
    DIR *dir = opendir(testDir);
    if (!dir) {
        printf("Error: Cannot open directory %s\n", testDir);
        return -1;
    }
    
    std::vector<std::string> imageFiles;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL) {
        std::string filename = entry->d_name;
        if (filename.find(".jpg") != std::string::npos ||
            filename.find(".jpeg") != std::string::npos ||
            filename.find(".png") != std::string::npos) {
            std::string fullpath = std::string(testDir) + "/" + filename;
            imageFiles.push_back(fullpath);
        }
    }
    closedir(dir);
    
    if (imageFiles.empty()) {
        printf("Error: No test images found\n");
        return -1;
    }
    
    printf("Found %zu test images\n\n", imageFiles.size());
    
    std::map<std::string, std::map<std::string, int>> confusionMatrix;
    std::set<std::string> uniqueLabelsSet;
    for (const auto &filepath : imageFiles) {
        std::string label = extractLabelFromFilename(filepath);
        uniqueLabelsSet.insert(label);
    }
    std::vector<std::string> allLabels(uniqueLabelsSet.begin(), uniqueLabelsSet.end());
    std::sort(allLabels.begin(), allLabels.end());

    printf("Detected object classes: ");
    for (const auto &label : allLabels) {
        printf("%s ", label.c_str());
    }
    printf("\n\n");
    
    for (const auto &label : allLabels) {
        for (const auto &pred : allLabels) {
            confusionMatrix[label][pred] = 0;
        }
        confusionMatrix[label]["unknown"] = 0;
    }
    
    int totalTests = 0;
    int correct = 0;
    
    for (const auto &filepath : imageFiles) {
        std::string trueLabel = extractLabelFromFilename(filepath);
        
        cv::Mat image = cv::imread(filepath);
        if (image.empty()) {
            printf("Warning: Could not read %s\n", filepath.c_str());
            continue;
        }
        
        cv::Mat thresholded, cleaned, labels;
        std::vector<RegionStats> stats;
        std::vector<RegionFeatures> features;
        
        preprocessAndThreshold(image, thresholded);
        cleanupBinary(thresholded, cleaned);
        
        int numRegions = segmentRegions(cleaned, labels, stats, 1000);
        
        if (numRegions == 0) {
            printf("No regions in %s\n", filepath.c_str());
            confusionMatrix[trueLabel]["unknown"]++;
            totalTests++;
            continue;
        }
        
        computeAllFeatures(labels, stats, features);
        
        if (features.empty()) {
            confusionMatrix[trueLabel]["unknown"]++;
            totalTests++;
            continue;
        }
        
        std::vector<RegionStats> largest = getLargestRegions(stats, 1);
        RegionFeatures mainFeature;
        bool found = false;
        
        for (const auto &feat : features) {
            if (feat.label == largest[0].label) {
                mainFeature = feat;
                found = true;
                break;
            }
        }
        
        if (!found) {
            confusionMatrix[trueLabel]["unknown"]++;
            totalTests++;
            continue;
        }
        
        ClassificationResult result = classifyObject(mainFeature, database, stdevs, 1, 2.0);
        
        confusionMatrix[trueLabel][result.label]++;
        totalTests++;
        
        if (result.label == trueLabel && !result.isUnknown) {
            correct++;
        }
        
        printf("%-15s ->    %-15s (dist: %.3f)\n", trueLabel.c_str(), result.label.c_str(), result.distance);
    }
    
    printf("\n=== Confusion Matrix ===\n");
    printf("%-15s", "True\\Pred");
    for (const auto &label : allLabels) {
        printf("%-15s", label.c_str());
    }
    printf("%-15s\n", "unknown");

    for (size_t i = 0; i < (allLabels.size() + 1) * 15 + 15; i++) {
        printf("-");
    }
    printf("\n");

    for (const auto &trueLabel : allLabels) {
        printf("%-15s", trueLabel.c_str());
        for (const auto &predLabel : allLabels) {
            printf("%-15d", confusionMatrix[trueLabel][predLabel]);
        }
        printf("%-15d\n", confusionMatrix[trueLabel]["unknown"]);
    }

    // Save to CSV
    std::ofstream csvFile("confusion_matrix_handcrafted.csv");
    if (csvFile.is_open()) {
        csvFile << "True\\Pred";
        for (const auto &label : allLabels) {
            csvFile << "," << label;
        }
        csvFile << ",unknown\n";
        
        for (const auto &trueLabel : allLabels) {
            csvFile << trueLabel;
            for (const auto &predLabel : allLabels) {
                csvFile << "," << confusionMatrix[trueLabel][predLabel];
            }
            csvFile << "," << confusionMatrix[trueLabel]["unknown"] << "\n";
        }
        csvFile.close();
        printf("\nConfusion matrix saved to: confusion_matrix_handcrafted.csv\n");
    }
    
    printf("\n=== Evaluation Results ===\n");
    printf("Total test images: %d\n", totalTests);
    printf("Correct classifications: %d\n", correct);
    printf("Accuracy: %.2f%%\n", 100.0 * correct / totalTests);
    
    return 0;
}