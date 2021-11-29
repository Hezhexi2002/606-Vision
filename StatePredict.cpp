#include <opencv2/core/types.hpp>
#include <iostream>
#include "StatePredict.hpp"
//打开xml文件需要加载的头文件
//#include "tinystr.h"  
//#include "tinyxml.h"

#define CAMERA_CFG "./camera_params.xml"

using namespace std;
using namespace cv;



namespace NUC606
{
    StatePredict::StatePredict()
    {
   
        m_KF = std::make_shared<cv::KalmanFilter>(m_state_num, m_measure_num, 0);
        measurement = cv::Mat::zeros(m_measure_num, 1, CV_32F);
        cv::setIdentity(m_KF->measurementMatrix);
        cv::setIdentity(m_KF->processNoiseCov, cv::Scalar::all(1e-5));
        cv::setIdentity(m_KF->measurementNoiseCov, cv::Scalar::all(1e-3));
        cv::setIdentity(m_KF->errorCovPost, cv::Scalar::all(1));
        float dt = 1.f / m_freq;
        m_KF->transitionMatrix = (cv::Mat_<float>(m_state_num, m_state_num) <<
            1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0,
            0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0,
            0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt,
            0, 0, 0, 1, 0, 0, dt, 0, 0,
            0, 0, 0, 0, 1, 0, 0, dt, 0,
            0, 0, 0, 0, 0, 1, 0, 0, dt,
            0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1);
    }


    void StatePredict::ReadParam() //读取相机参数 （待读取）
    {
        cv::FileStorage file("./camera_params.xml", cv::FileStorage::READ);
       
    
    
    
    
    }


    void StatePredict::InitFilter(cv::Point3f a_position)
    {
        m_KF->statePost = (cv::Mat_<float>(m_state_num, 1) <<
            a_position.x, a_position.y, a_position.z, 0, 0, 0, 0, 0, 0);
        m_KF->predict();
        measurement.at<float>(0) = a_position.x;
        measurement.at<float>(1) = a_position.y;
        measurement.at<float>(2) = a_position.z;

        cur_armor_dis = std::sqrt(std::pow(a_position.x, 2) + std::pow(a_position.y, 2) + std::pow(a_position.z, 2));
        last_armor_dis = std::sqrt(std::pow(a_position.x, 2) + std::pow(a_position.y, 2) + std::pow(a_position.z, 2));

        for (int i = 0; i < init_count_threshold; i++)  // 对第一帧进行多次预测
        {
            m_KF->correct(measurement);
            m_KF->predict();
        }    
    
    }







    cv::Point3f StatePredict::Predict(cv::Point3f a_position)
    {
        cv::Mat prediction;
     //   cur_armor_dis = std::sqrt(std::pow(a_position.x, 2) + std::pow(a_position.y, 2) + std::pow(a_position.z, 2));

        if (!m_init)
        {
            std::cout << "Init..." << std::endl;
            InitFilter(a_position);
            x_k1 = a_position;
            m_init = true;
        }

        measurement.at<float>(0) = a_position.x;
        measurement.at<float>(1) = a_position.y;
        measurement.at<float>(2) = a_position.z;
        m_KF->correct(measurement);
        prediction = m_KF->predict();
       /* if (std::sqrt(std::pow(std::fabs(a_position.x - x_k1.x), 2) + std::pow(std::fabs(a_position.y - x_k1.y), 2) + std::pow(std::fabs(a_position.z - x_k1.z), 2)) > m_target_change_threshold)
        {
            InitFilter(a_position);
            if (m_debug)
                std::cout << "Target has changed" << std::endl;

        }*/
        x_k1 = a_position;
      //  last_armor_dis = cur_armor_dis;

        return cv::Point3f(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
    }

}



