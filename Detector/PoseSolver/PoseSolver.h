#pragma once
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include<iostream>

enum ArmorType
{
	smallArmor = 0,
	bigArmor = 1
};

struct CalculateResults
{
	double roll, pitch, yaw, distance;
};

class PoseSolver
{
public:
	
	PoseSolver();

	~PoseSolver();

	CalculateResults calResults;

	struct CalculateResults* calResult = (CalculateResults*)malloc(sizeof(calResults));//声明结构体指针变量，并为其分配内存空间

	struct CalculateResults* PnPSolve();	//PNP解算

	void setCameraParams(const cv::Mat& camMatrix, const cv::Mat& distCoeffs);//设定相机参数

	int readFile(const char* filePath, int camId);//从xml文件读取相机参数

	void setObjPoints(ArmorType type, double width, double height);//设置装甲板世界坐标

	void getImgpPoints(cv::Rect2f& rect);//获得装甲板相机坐标

	void gravityCompensation();

	void poseCompensation();
	
	void pinholeSolve();

	double poseSolve();

private:

	cv::Mat instantMatrix;	//Camera Matrix
	cv::Mat distortionCoeffs;	//Distortion Coeffs of Camera

	std::vector<cv::Point3f> bigObjPoints;	//object points of big arrmors
	std::vector<cv::Point3f> smallObjPoints;	//object points of small arrmors
	
	std::vector<cv::Point2f> imagePoints;	//image_points of arrmors
	
	
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);	//旋转向量
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);	//平移向量

	std::float speed;


};


