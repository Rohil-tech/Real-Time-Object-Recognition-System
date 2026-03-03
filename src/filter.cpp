/**
 * Rohil Kulshreshtha
 * February 11, 2026
 * CS 5330 - PR-CV - Assignment 3
 * 
 * Image filtering operations (from Assignment 1)
 */

#include <opencv2/opencv.hpp>
#include "filter.h"

int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    cv::Mat tmp;
    src.copyTo(tmp);

    // Horizontal pass: [1 2 4 2 1]
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tptr = tmp.ptr<cv::Vec3b>(i);
        
        for(int j = 2; j < src.cols - 2; j++) {
            for(int c = 0; c < 3; c++) {
                int value = sptr[j-2][c] * 1 + sptr[j-1][c] * 2 + sptr[j][c] * 4 + sptr[j+1][c] * 2 + sptr[j+2][c] * 1;
                tptr[j][c] = value / 10;
            }
        }
    }

    tmp.copyTo(dst);

    // Vertical pass: [1 2 4 2 1]^T
    for(int i = 2; i < src.rows - 2; i++) {
        cv::Vec3b *tptr_up2 = tmp.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *tptr_up1 = tmp.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *tptr_mid = tmp.ptr<cv::Vec3b>(i);
        cv::Vec3b *tptr_dn1 = tmp.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *tptr_dn2 = tmp.ptr<cv::Vec3b>(i+2);
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
        
        for(int j = 0; j < src.cols; j++) {
            for(int c = 0; c < 3; c++) {
                int value = tptr_up2[j][c] * 1 + tptr_up1[j][c] * 2 + tptr_mid[j][c] * 4 + tptr_dn1[j][c] * 2 + tptr_dn2[j][c] * 1;
                dptr[j][c] = value / 10;
            }
        }
    }

    return 0;
}