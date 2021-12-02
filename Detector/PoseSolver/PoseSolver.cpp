#include "cv.h"
#include "highgui.h"
#include "math.h"
#include <iostream>
#include <string>
#include <windows.h>
using namespace std;

void main()
{
    IplImage* src;
    IplImage* dst;
    WIN32_FIND_DATA p; //指向一个用于保存文件信息的结构体
    HANDLE h = FindFirstFile("C:\\Users\\23937\\Desktop\\datas\\data*.jpg", &p); //FindFirstFile的返回值是一个句柄，第二个参数p是采用引用的方式，也就是说当这句话执行完毕后p就指向该文件*.jpg

    //由于p的成员变量只有文件名，而无文件路径，所以必须加上路径头
    string src_route_head = "C:\\Users\\23937\\Desktop\\datas\\data\\"; //源图像的路径头
    string dst_route_head = "C:\\Users\\23937\\Desktop\\datas\\newdata\\";//目标图像的路径头
    string SourceRoute =src_route_head + p.cFileName; //包含了路径头和文件名的全路径
    string DestRoute = dst_route_head + p.cFileName;

    src = cvLoadImage(SourceRoute.c_str(), 0);//载入源图像
    dst = cvCreateImage(cvSize(256, 256), src->depth, src->nChannels);//分配一个256*256的目标图像，resize后的结果将放在这里

    cvResize(src, dst);
    cvSaveImage(DestRoute.c_str(), dst);//保存dst

    //到目前为止，我们就已经完成了对目标文件夹中第一幅图像的resize处理与保存，接下来让该文件夹中其余图像也被处理

    while (FindNextFile(h, &p)) //p指针不断后移，寻找下一个、下下一个*.jpg
    {
        SourceRoute = src_route_head + p.cFileName; //route是路径头，imageroute是包含了路径头和文件名的全路径
        src = cvLoadImage(SourceRoute.c_str(), 0);//载入源图像

        cvResize(src, dst);

        DestRoute = dst_route_head + p.cFileName;
        cvSaveImage(DestRoute.c_str(), dst);//保存dst
    }
}
























//#include "PoseSolver.h"
//#include <opencv2/opencv.hpp>
//#include<iostream>
//using namespace cv;
//using namespace std;
//
//PoseSolver::PoseSolver(void)
//{
//}
//
//PoseSolver::~PoseSolver(void)
//{
//	if (calResult != NULL)
//	{
//		free(calResult);
//	}
//}
//
//void PoseSolver::setCameraParams(const Mat& camMatrix, const Mat& distCoeffs)		//设置相机参数
//{
//	instantMatrix = camMatrix;
//	distortionCoeffs = distCoeffs;
//}
//
//int PoseSolver::readFile(const char* filePath, int camId)		//读取参数文件
//{
//	FileStorage fsRead;
//	fsRead.open(filePath, FileStorage::READ);
//	if (!fsRead.isOpened())
//	{
//		cout << "Failed to open xml" << endl;
//		return -1;
//	}
//
//	//fsRead["Y_DISTANCE_BETWEEN_GUN_AND_CAM"] >> GUN_CAM_DISTANCE_Y;
//
//	Mat cameraMatrix;
//	Mat distortionCoeffs;
//	switch (camId)
//	{
//	case 1:
//		fsRead["CAMERA_MATRIX_1"] >> cameraMatrix;
//		fsRead["DISTORTION_COEFF_1"] >> distortionCoeffs;
//		break;
//	default:
//		cout << "WRONG CAMID GIVEN!" << endl;
//		break;
//	}
//	setCameraParams(cameraMatrix, distortionCoeffs);
//	fsRead.release();
//	return 0;
//}
//
//void PoseSolver::gravityCompensation()		//重力补偿
//{
//	if (calResult->distance > 0 && calResult->distance < 3000)
//	{
//		cout << 1000;
//	}
//	else if (calResult->distance > 3000 && calResult->distance < 5000)
//	{
//		cout << 1000;
//	}
//}
//
//void PoseSolver::poseCompensation()		//位姿补偿
//{
//
//}
//
//struct* PoseSolver::PnPSolve()		//PNP解算
//{
//	//struct CalculateResults* calResult = (CalculateResults*)malloc(sizeof(calResults));//声明结构体指针变量，并为其分配内存空间
//	solvePnP(smallObjPoints, imagePoints, instantMatrix, distortionCoeffs, rvec, tvec, false, CV_ITERATIVE);
//	Mat rotM = Mat::eye(3, 3, CV_64F);
//	Mat rotT = Mat::eye(3, 3, CV_64F);
//	Rodrigues(rvec, rotM);  //将旋转向量变换成旋转矩阵
//	Rodrigues(tvec, rotT);  //将平移向量变换成旋转矩阵
//	double thetaX, thetaY, thetaZ;
//	double PI = 3.14;
//	thetaX = atan2(rotM.at<double>(2, 1), rotM.at<double>(2, 2));
//	thetaY = atan2(-rotM.at<double>(2, 0),
//		sqrt(rotM.at<double>(2, 0) * rotM.at<double>(2, 0) + rotM.at<double>(2, 2) * rotM.at<double>(2, 2)));
//	thetaZ = atan2(rotM.at<double>(1, 0), rotM.at<double>(0, 0));
//	calResult->pitch = thetaX * (180 / PI);
//	calResult->yaw = thetaY * (180 / PI);
//	Mat P;
//	P = -(rotM.t()) * tvec;
//	calResult->distance = P.at<double>(2, 0);
//	return calResult;
//}
//
//void PoseSolver::setObjPoints(ArmorType type, double width, double height)		//设置装甲板世界坐标系角点
//{
//	double centerX = width / 2.0;
//	double centerY = height / 2.0;
//	switch (type)
//	{
//	case smallArmor:
//		smallObjPoints.push_back(Point3f(-centerX, centerY, 0));   //tl top left左上
//		smallObjPoints.push_back(Point3f(centerX, centerY, 0));	//tr top right右上
//		smallObjPoints.push_back(Point3f(centerX, -centerY, 0));   //br below right右下
//		smallObjPoints.push_back(Point3f(-centerX, -centerY, 0));  //bl below left左下
//		break;
//
//	case bigArmor:
//		bigObjPoints.push_back(Point3f(centerX, -centerY, 0));   //tl top left左上
//		bigObjPoints.push_back(Point3f(centerX, centerY, 0));    //tr top right右上
//		bigObjPoints.push_back(Point3f(-centerX, centerY, 0));   //bl below left右下
//		bigObjPoints.push_back(Point3f(-centerX, -centerY, 0));  //br below right左下
//		break;
//	default: break;
//	}
//}
//
//void PoseSolver::getImgpPoints(Rect2f& rect)		//获取图像坐标角点
//{
//	imagePoints.clear();
//	imagePoints.push_back(rect.tl());
//	imagePoints.push_back(Point2f(rect.tl().x + rect.width, rect.tl().y));
//	imagePoints.push_back(rect.br());
//	imagePoints.push_back(Point2f(rect.br().x - rect.width, rect.br().y));
//}
//
//vodi PoseSolver::pinholeSolve()		//小孔成像解算位姿
//{
//	double fx = instantMatrix.at<double>(0, 0);
//	double fy = instantMatrix.at<double>(1, 1);
//	double cx = instantMatrix.at<double>(0, 2);
//	double cy = instantMatrix.at<double>(1, 2);
//	Point2f pnt;
//	vector<cv::Point2f> inImg;
//	vector<cv::Point2f> outImg;
//	in.push_back(targetCenter);
//
//	//对像素点去畸变
//	undistortPoints(inImg, outImg, instantMatrix, distortionCoeffs, noArray(), CAMERA_MATRIX);
//	pnt = outImg.front();
//
//	//去畸变后的比值
//	double rxNew = (pnt.x - cx) / fx;
//	double ryNew = (pnt.y - cy) / fy;
//
//	//y_yaw = atan(rxNew) / CV_PI * 180;
//	//x_pitch = -atan(ryNew) / CV_PI * 180;
//}
//
//double PoseSolver::poseSolve()
//{
//	if (1)
//	{
//		PnPSolve();
//	}
//	else if (2)
//	{
//		pinholeSolve();
//	}
//}
