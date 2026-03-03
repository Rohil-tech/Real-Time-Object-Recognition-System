/**
 * Rohil Kulshreshtha
 * February 18, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Batch training program to build object database from images.
 */

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include "threshold.h"
#include "morphology.h"
#include "segmentation.h"
#include "features.h"
#include "database.h"

int main(int argc, char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    if (argc < 3) {
        printf("Usage: %s <image_directory> <output_csv>\n", argv[0]);
        printf("Example: %s data/training_images data/object_database.csv\n", argv[0]);
        return -1;
    }
    
    const char *dirname = argv[1];
    const char *outputCSV = argv[2];
    
    printf("=== Training Object Recognition System ===\n");
    printf("Input directory: %s\n", dirname);
    printf("Output database: %s\n\n", outputCSV);
    
    DIR *dir = opendir(dirname);
    if (!dir) {
        printf("Error: Cannot open directory %s\n", dirname);
        return -1;
    }
    
    std::vector<std::string> imageFiles;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL) {
        std::string filename = entry->d_name;
        if (filename.find(".jpg") != std::string::npos ||
            filename.find(".jpeg") != std::string::npos ||
            filename.find(".png") != std::string::npos) {
            std::string fullpath = std::string(dirname) + "/" + filename;
            imageFiles.push_back(fullpath);
        }
    }
    closedir(dir);
    
    if (imageFiles.empty()) {
        printf("Error: No images found in directory\n");
        return -1;
    }
    
    printf("Found %zu images\n\n", imageFiles.size());
    
    bool firstEntry = true;
    int successCount = 0;
    int failCount = 0;
    
    for (size_t i = 0; i < imageFiles.size(); i++) {
        std::string filepath = imageFiles[i];
        std::string label = extractLabelFromFilename(filepath);
        
        cv::Mat image = cv::imread(filepath);
        if (image.empty()) {
            printf("Warning: Could not read %s\n", filepath.c_str());
            failCount++;
            continue;
        }
        
        cv::Mat thresholded, resized, cleaned, labels;
        std::vector<RegionStats> stats;
        std::vector<RegionFeatures> features;

        int maxDim = image.cols > image.rows ? image.cols : image.rows;
        if (maxDim > 800) {
            double scale = 800.0 / maxDim;
            int newWidth = image.cols * scale;
            int newHeight = image.rows * scale;
            cv::resize(image, resized, cv::Size(newWidth, newHeight));
        } else {
            resized = image;
        }
        
        preprocessAndThreshold(resized, thresholded);
        cleanupBinary(thresholded, cleaned);
        
        int numRegions = segmentRegions(cleaned, labels, stats, 100);
        
        if (numRegions == 0) {
            printf("Warning: No regions found in %s\n", filepath.c_str());
            failCount++;
            continue;
        }
        
        computeAllFeatures(labels, stats, features);
        
        if (features.empty()) {
            printf("Warning: No features computed for %s\n", filepath.c_str());
            failCount++;
            continue;
        }
        
        RegionFeatures &feat = features[0];
        if (features.size() > 1) {
            std::vector<RegionStats> largest = getLargestRegions(stats, 1);
            for (size_t j = 0; j < features.size(); j++) {
                if (features[j].label == largest[0].label) {
                    feat = features[j];
                    break;
                }
            }
        }
        
        saveFeatureToDatabase(outputCSV, label.c_str(), feat, !firstEntry);
        firstEntry = false;
        successCount++;
        
        if ((i + 1) % 10 == 0) {
            printf("Processed %zu/%zu images...\n", i + 1, imageFiles.size());
        }
    }
    
    printf("\n=== Training Complete ===\n");
    printf("Successfully processed: %d images\n", successCount);
    printf("Failed: %d images\n", failCount);
    printf("Database saved to: %s\n", outputCSV);
    
    return 0;
}