/**
 * Rohil Kulshreshtha
 * February 19, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * CNN embedding extraction using ResNet18.
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "features.h"

/**
 * Prepare ROI image for embedding extraction.
 * Rotates image to align primary axis, extracts region, resizes to square.
 * 
 * @param frame Original image
 * @param features Region features (contains centroid, orientation, extents)
 * @param roiImage Output extracted and rotated ROI
 * @param targetSize Output size (default: 224 for ResNet)
 * @return 0 on success
 */
int prepareROI(const cv::Mat &frame, const RegionFeatures &features, cv::Mat &roiImage, int targetSize = 224);

/**
 * Extract embedding from image using ResNet18.
 * 
 * @param roiImage Input image (224x224)
 * @param embedding Output embedding vector
 * @param net ResNet18 network
 * @return 0 on success
 */
int getEmbedding(const cv::Mat &roiImage, cv::Mat &embedding, cv::dnn::Net &net);

/**
 * Load ResNet18 network from ONNX file.
 * 
 * @param modelPath Path to resnet18-v2-7.onnx
 * @param net Output network
 * @return 0 on success
 */
int loadResNet18(const char *modelPath, cv::dnn::Net &net);

/**
 * Compute distance between two embeddings (Sum of Squared Differences).
 * 
 * @param emb1 First embedding
 * @param emb2 Second embedding
 * @return SSD distance
 */
double embeddingDistance(const cv::Mat &emb1, const cv::Mat &emb2);

#endif // EMBEDDING_H