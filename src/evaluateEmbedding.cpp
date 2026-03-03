/**
 * Rohil Kulshreshtha
 * February 20, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Evaluate embedding-based classification and compare with hand-crafted features.
 */

#include <cstdio>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include <map>
#include "threshold.h"
#include "morphology.h"
#include "segmentation.h"
#include "features.h"
#include "database.h"
#include "embedding.h"

struct EmbeddingEntry {
    std::string label;
    cv::Mat embedding;
};

int loadEmbeddingDatabase(const char *filename, std::vector<EmbeddingEntry> &entries) {
    entries.clear();
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        return -1;
    }
    
    std::string line;
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        EmbeddingEntry entry;
        
        std::getline(ss, entry.label, ',');
        
        std::vector<float> values;
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stof(token));
        }
        
        entry.embedding = cv::Mat(1, values.size(), CV_32F);
        for (size_t i = 0; i < values.size(); i++) {
            entry.embedding.at<float>(i) = values[i];
        }
        
        entries.push_back(entry);
    }
    
    file.close();
    return entries.size();
}

std::string classifyWithEmbedding(const cv::Mat &embedding, const std::vector<EmbeddingEntry> &database) {
    if (database.empty()) {
        return "unknown";
    }
    
    double minDist = 999999.0;
    std::string bestLabel = "unknown";
    
    for (const auto &entry : database) {
        double dist = embeddingDistance(embedding, entry.embedding);
        if (dist < minDist) {
            minDist = dist;
            bestLabel = entry.label;
        }
    }
    
    return bestLabel;
}

int main(int argc, char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    if (argc < 4) {
        printf("Usage: %s <test_dir> <model_path> <embedding_csv>\n", argv[0]);
        return -1;
    }
    
    const char *testDir = argv[1];
    const char *modelPath = argv[2];
    const char *embeddingCSV = argv[3];
    
    printf("=== Evaluating Embedding-Based Classification ===\n\n");
    
    cv::dnn::Net net;
    if (loadResNet18(modelPath, net) != 0) {
        return -1;
    }
    
    std::vector<EmbeddingEntry> database;
    int loaded = loadEmbeddingDatabase(embeddingCSV, database);
    if (loaded <= 0) {
        printf("Error: Could not load embedding database\n");
        return -1;
    }
    printf("Loaded %d embeddings from database\n\n", loaded);
    
    DIR *dir = opendir(testDir);
    if (!dir) {
        printf("Error: Cannot open test directory\n");
        return -1;
    }
    
    std::vector<std::string> imageFiles;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL) {
        std::string filename = entry->d_name;
        if (filename.find(".jpg") != std::string::npos ||
            filename.find(".jpeg") != std::string::npos ||
            filename.find(".png") != std::string::npos) {
            imageFiles.push_back(std::string(testDir) + "/" + filename);
        }
    }
    closedir(dir);
    
    printf("Found %zu test images\n\n", imageFiles.size());
    
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
    
    std::map<std::string, std::map<std::string, int>> confusionMatrix;
    for (const auto &trueLabel : allLabels) {
        for (const auto &pred : allLabels) {
            confusionMatrix[trueLabel][pred] = 0;
        }
    }
    
    int totalTests = 0;
    int correct = 0;
    
    for (const auto &filepath : imageFiles) {
        std::string trueLabel = extractLabelFromFilename(filepath);
        
        cv::Mat image = cv::imread(filepath);
        if (image.empty()) continue;
        
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
        segmentRegions(cleaned, labels, stats, 100);
        computeAllFeatures(labels, stats, features);
        
        if (features.empty()) continue;
        
        std::vector<RegionStats> largest = getLargestRegions(stats, 1);
        RegionFeatures feat;
        bool found = false;
        for (const auto &f : features) {
            if (f.label == largest[0].label) {
                feat = f;
                found = true;
                break;
            }
        }
        
        if (!found) continue;
        
        cv::Mat roiImage, embedding;
        if (prepareROI(resized, feat, roiImage) != 0) continue;
        if (getEmbedding(roiImage, embedding, net) != 0) continue;
        
        std::string predLabel = classifyWithEmbedding(embedding, database);
        
        confusionMatrix[trueLabel][predLabel]++;
        totalTests++;
        
        printf("%-15s ->    %-15s\n", trueLabel.c_str(), predLabel.c_str());
        if (predLabel == trueLabel) {
            correct++;
        }
    }
    
    printf("\n=== Confusion Matrix (Embeddings) ===\n");
    printf("True \\ Pred  ");
    for (const auto &label : allLabels) {
        printf("%-12s ", label.c_str());
    }
    printf("\n");
    
    for (const auto &trueLabel : allLabels) {
        printf("%-12s  ", trueLabel.c_str());
        for (const auto &predLabel : allLabels) {
            printf("%-12d ", confusionMatrix[trueLabel][predLabel]);
        }
        printf("\n");
    }
    
    printf("\n=== Embedding Results ===\n");
    printf("Total: %d\n", totalTests);
    printf("Correct: %d\n", correct);
    printf("Accuracy: %.2f%%\n", 100.0 * correct / totalTests);
    
    return 0;
}