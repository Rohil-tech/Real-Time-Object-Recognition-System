/**
 * Rohil Kulshreshtha
 * February 16, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Database management implementation
 */

#include <fstream>
#include <sstream>
#include <cstring>
#include "database.h"

int saveFeatureToDatabase(const char *filename, const char *label, const RegionFeatures &features, bool append) {
    std::ofstream file;
    
    if (append) {
        // Check if file exists first
        std::ifstream checkFile(filename);
        bool fileExists = checkFile.good();
        checkFile.close();
        
        if (!fileExists) {
            // File doesn't exist, create it with header
            file.open(filename, std::ios::out);
            if (!file.is_open()) {
                printf("Error: Could not create %s\n", filename);
                return -1;
            }
            file << "label,percentFilled,aspectRatio,huMoment1,huMoment2\n";
        } else {
            // File exists, append
            file.open(filename, std::ios::app);
            if (!file.is_open()) {
                printf("Error: Could not open %s for appending\n", filename);
                return -1;
            }
        }
    } else {
        // Create new file with header
        file.open(filename, std::ios::out);
        if (!file.is_open()) {
            printf("Error: Could not create %s\n", filename);
            return -1;
        }
        file << "label,percentFilled,aspectRatio,huMoment1,huMoment2\n";
    }
    
    file << label << ","
         << features.percentFilled << ","
         << features.aspectRatio << ","
         << features.huMoments[0] << ","
         << features.huMoments[1] << "\n";
    
    file.close();
    return 0;
}

int loadDatabase(const char *filename, std::vector<DatabaseEntry> &entries) {
    entries.clear();
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open %s\n", filename);
        return -1;
    }
    
    std::string line;
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        DatabaseEntry entry;
        
        std::getline(ss, entry.label, ',');
        
        std::getline(ss, token, ',');
        entry.features.percentFilled = std::stod(token);
        
        std::getline(ss, token, ',');
        entry.features.aspectRatio = std::stod(token);
        
        std::getline(ss, token, ',');
        entry.features.huMoments[0] = std::stod(token);
        
        std::getline(ss, token, ',');
        entry.features.huMoments[1] = std::stod(token);
        
        entries.push_back(entry);
    }
    
    file.close();
    return entries.size();
}

std::string extractLabelFromFilename(const std::string &filename) {
    size_t lastSlash = filename.find_last_of("/\\");
    std::string basename = (lastSlash != std::string::npos) ? filename.substr(lastSlash + 1) : filename;
    
    size_t lastUnderscore = basename.find_last_of("_");
    if (lastUnderscore != std::string::npos) {
        return basename.substr(0, lastUnderscore);
    }
    
    size_t lastDot = basename.find_last_of(".");
    if (lastDot != std::string::npos) {
        return basename.substr(0, lastDot);
    }
    
    return basename;
}