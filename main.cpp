#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/types_c.h>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

void trackObject(const Mat* treshImage, const Mat* imageTracing) 
{
	vector < vector<Point> > contours;

	findContours(*treshImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	size_t size = contours.size();

	vector < vector<Point> > result;
	result.resize(size);

	for (size_t i = 0; i < size; ++i) {

		approxPolyDP(contours[i], result[i], arcLength(contours[i], true)*0.02, true);

		if (result[i].size() == 4 && fabs(contourArea(contours[i], false)) > 100) {
			
			vector<Point> points = result[i];
			size_t size2D = points.size();

			line(*imageTracing, points[0], points[1], Scalar(0, 255, 0), 3, LINE_AA, 0);
			line(*imageTracing, points[1], points[2], Scalar(0, 255, 0), 3, LINE_AA, 0);
			line(*imageTracing, points[2], points[3], Scalar(0, 255, 0), 3, LINE_AA, 0);
			line(*imageTracing, points[3], points[0], Scalar(0, 255, 0), 3, LINE_AA, 0);
			//line(*imageTracing, points[4], points[5], Scalar(0, 255, 0), 3, LINE_AA, 0);
			//line(*imageTracing, points[5], points[6], Scalar(0, 255, 0), 3, LINE_AA, 0);
			//line(*imageTracing, points[6], points[0], Scalar(0, 255, 0), 3, LINE_AA, 0);

			vector<Point2d> points2D;
			for (size_t i = 0; i < size2D; ++i) {
				
				points2D.push_back(Point2d(points[i].x, points[i].y));
			}

			vector<Point3d> points3D;
			for (size_t i = 0; i < size2D; ++i) {
				
				points3D.push_back(Point3d(points2D[i].x, points2D[i].y, 0));
			}

			const double focalLength = treshImage->cols; // camera parameters
			Point2d center(treshImage->cols / 2, treshImage->rows / 2);
			double cameraParam[4];
			cameraParam[0] = focalLength;
			cameraParam[1] = center.x;
			cameraParam[2] = focalLength;
			cameraParam[3] = center.y;

			Mat cameraMatrix = Mat::zeros(3, 3, CV_64FC1);  // intrisinc camera parameters
			cameraMatrix.at<double>(0, 0) = cameraParam[0]; // [ fx 0 cx ]
			cameraMatrix.at<double>(0, 2) = cameraParam[1]; // [ 0 fy cy ]
			cameraMatrix.at<double>(1, 1) = cameraParam[2]; // [ 0  0  1 ]
			cameraMatrix.at<double>(1, 2) = cameraParam[3];
			cameraMatrix.at<double>(2, 2) = 1;


			Mat distCoeffs = Mat::zeros(4, 1, DataType<double>::type);
			Mat rvec;
			Mat tvec;
			
			solvePnP(points3D, points2D, cameraMatrix, distCoeffs, rvec, tvec);

			cout << "Rotation vector: " << rvec << endl;
			cout << "Translation vector: " << tvec << endl;
		}
	}
}


int main(int argc, char* argv[]) 
{
	VideoCapture cap(0); // 0 means default camera
	if (!cap.isOpened()) {
		
		printf("Can`t open the video file!\n");
		return -1;
	}

	namedWindow("Video");

	Mat frame;
	if (!cap.read(frame)) {

		printf("Can`t read frame!\n");
		return -2;
	}

	while (cap.read(frame) && waitKey(10) != 27) {

		Mat imageTracing = Mat::zeros(frame.rows, frame.cols, frame.type());
		
		GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);
		
		Mat grayScaleImage;
		cvtColor(frame, grayScaleImage, CV_BGR2GRAY);

		threshold(grayScaleImage, grayScaleImage, 100, 255, CV_THRESH_BINARY_INV);

		trackObject(&grayScaleImage, &imageTracing);

		add(frame, imageTracing, frame);

		imshow("Video", frame);
	}

	destroyWindow("Video");

	return 0;
}