#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <thread>
#include <opencv2/core/types.hpp>

namespace NUC606
{
	// 三维下的预测
	class StatePredict
	{
	public:
		StatePredict() = default;     /*构造函数*/

		StatePredict(cv::Mat A, cv::Mat H, cv::Mat R, cv::Mat Q, cv::Mat P, cv::Mat x_k1);

		~StatePredict() {}  // 析构函数

		void InitFilter(cv::Point3f a_position); // 初始化滤波器

		void ReadParam(); // 读取一些参数

		cv::Point3f Predict(cv::Point3f); 

	private:

		std::shared_ptr<cv::KalmanFilter> m_KF;

		int m_freq; 
		bool m_init = false;
		int m_init_count_max;
		float m_dt;  // 状态转换矩阵里的时间变量
		int m_measure_num = 3; //观测矩阵中元素个数 
		int m_state_num = 9;  //状态矩阵中元素个数 每一个方向的位置p，速度v，加速度a 故3*3=9
		

		/*****************************************************/
		/*需要读取的参数*/
		int init_count_threshold;  //对于第一帧所预测的次数 （多次预测取最优值）
		float cur_armor_dis;  // 当前装甲板的距离
		float last_armor_dis; // 上一刻装甲板的距离
		int m_debug;
		cv::Mat R_CI_MAT;               // 陀螺仪坐标系到相机坐标系旋转矩阵CV-Mat
		cv::Mat F_MAT;                  // 相机内参矩阵CV-Mat
		cv::Mat C_MAT;                  // 相机畸变矩阵CV-Mat
		/*****************************************************/

		cv::Mat measurement;    // 观测矩阵
		cv::Mat x_k;            // x_k时刻的值，后验值  现在时刻
		cv::Point3f x_k1;			// x_k-1时刻的值，先验值  上一时刻
		cv::Mat A;			  // A 状态转换矩阵
		cv::Mat H;			 // H 观测矩阵
		cv::Mat Q;			// Q 预测噪声
		cv::Mat R;			// R 观测噪声
		cv::Mat P;        // P 系统协方差
	

		
	};

}