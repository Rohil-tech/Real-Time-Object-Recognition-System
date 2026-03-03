/**
 * Rohil Kulshreshtha
 * February 16, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Database management for object features.
 */

#ifndef DATABASE_H
#define DATABASE_H

#include <string>
#include <vector>
#include "features.h"

/**
 * Database entry: label + features.
 */
struct DatabaseEntry {
    std::string label;
    RegionFeatures features;
};

/**
 * Save feature vector to CSV database.
 * 
 * @param filename CSV file path
 * @param label Object label
 * @param features Feature vector
 * @param append If true, append to file; if false, create new file with header
 * @return 0 on success
 */
int saveFeatureToDatabase(const char *filename, const char *label, const RegionFeatures &features, bool append = true);

/**
 * Load database from CSV file.
 * 
 * @param filename CSV file path
 * @param entries Output vector of database entries
 * @return Number of entries loaded, -1 on error
 */
int loadDatabase(const char *filename, std::vector<DatabaseEntry> &entries);

/**
 * Extract label from filename.
 * Extracts everything before the last underscore or number.
 * Examples: "scissors_01.jpg" -> "scissors"
 *           "earbud_case_05.jpg" -> "earbud_case"
 * 
 * @param filename Input filename
 * @return Extracted label
 */
std::string extractLabelFromFilename(const std::string &filename);

#endif // DATABASE_H