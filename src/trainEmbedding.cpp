/**
 * Rohil Kulshreshtha
 * February 20, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Build embedding database using ResNet18.
 */

#include <cstdio>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include "threshold.h"
#include "morphology.h"
#include "segmentation.h"
#include "features.h"
#include "database.h"
#include "embedding.h"

int main(int argc, char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    if (argc < 4) {
        printf("Usage: %s <image_directory> <model_path> <output_csv>\n", argv[0]);
        printf("Example: %s data/training_images bin/resnet18-v2-7.onnx data/embedding_database.csv\n", argv[0]);
        return -1;
    }
    
    const char *dirname = argv[1];
    const char *modelPath = argv[2];
    const char *outputCSV = argv[3];
    
    printf("=== Training Embedding-Based Recognition ===\n");
    printf("Input directory: %s\n", dirname);
    printf("Model: %s\n", modelPath);
    printf("Output: %s\n\n", outputCSV);
    
    cv::dnn::Net net;
    if (loadResNet18(modelPath, net) != 0) {
        printf("Error: Failed to load ResNet18 model\n");
        return -1;
    }
    
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
        printf("Error: No images found\n");
        return -1;
    }
    
    printf("Found %zu images\n\n", imageFiles.size());
    
    std::ofstream outFile(outputCSV);
    if (!outFile.is_open()) {
        printf("Error: Could not create output file\n");
        return -1;
    }
    
    outFile << "label";
    for (int i = 0; i < 512; i++) {
        outFile << ",emb" << i;
    }
    outFile << "\n";
    
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
        
        int maxDim = (image.cols > image.rows) ? image.cols : image.rows;
        cv::Mat resized;
        if (maxDim > 800) {
            double scale = 800.0 / maxDim;
            cv::resize(image, resized, cv::Size(image.cols * scale, image.rows * scale));
        } else {
            resized = image;
        }
        
        cv::Mat thresholded, cleaned, labels;
        std::vector<RegionStats> stats;
        std::vector<RegionFeatures> features;
        
        preprocessAndThreshold(resized, thresholded);
        cleanupBinary(thresholded, cleaned);
        
        int numRegions = segmentRegions(cleaned, labels, stats, 100);
        
        if (numRegions == 0) {
            failCount++;
            continue;
        }
        
        computeAllFeatures(labels, stats, features);
        
        if (features.empty()) {
            failCount++;
            continue;
        }
        
        RegionFeatures feat = features[0];
        if (features.size() > 1) {
            std::vector<RegionStats> largest = getLargestRegions(stats, 1);
            for (const auto &f : features) {
                if (f.label == largest[0].label) {
                    feat = f;
                    break;
                }
            }
        }
        
        cv::Mat roiImage, embedding;
        if (prepareROI(resized, feat, roiImage) != 0) {
            failCount++;
            continue;
        }
        
        if (getEmbedding(roiImage, embedding, net) != 0) {
            failCount++;
            continue;
        }
        
        outFile << label;
        for (int j = 0; j < embedding.total(); j++) {
            outFile << "," << embedding.at<float>(j);
        }
        outFile << "\n";
        
        successCount++;
        
        if ((i + 1) % 10 == 0) {
            printf("Processed %zu/%zu images...\n", i + 1, imageFiles.size());
        }
    }
    
    outFile.close();
    
    printf("\n=== Training Complete ===\n");
    printf("Successfully processed: %d images\n", successCount);
    printf("Failed: %d images\n", failCount);
    printf("Database saved to: %s\n", outputCSV);
    
    return 0;
}