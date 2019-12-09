
#pragma once
#include <opencv2\opencv.hpp>
using namespace cv;
void PlateDetect(Mat OriginalImg,Mat &resImage);
//void PlateDetect(char *file);
void RectifyImage(char *file);
void KmeansSeg(char *file);
void ShowVideo(char *file);
