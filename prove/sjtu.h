#ifndef _SJTU_H
#define _SJTU_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>
#include <inference_engine.hpp>
#include <fstream>
#include <iostream>

using namespace InferenceEngine;
using namespace std;
using namespace cv;


struct box_t
{
	
	cv::Point2f pts[4];
	int color_id; 
    int tag_id;
	float conf;

	std::string getName() const {
		static const std::string tag2name[] = { "G", "1", "2", "3", "4", "5", "O", "Bs", "Bb" };
		static const std::string color2name[] = { "B", "R", "N", "P" };
		return color2name[color_id] + tag2name[tag_id];
	}

	bool setByName(const std::string &name) {
		static const std::map<std::string, int> name2tag;
		static const std::map<std::string, int> name2color;

		if ((name2color.contains(&name[0]) && name2tag.contains(&name[1])) == 1) {
			color_id = name2color.at(&name[0]);
			tag_id = name2tag.at(&name[1]);
			return true;
		}
		else {
			return false;
		}
	}

	std::vector<cv::Point2f> getStandardPloygon() const {
		std::vector<cv::Point2f> pts;
		pts.push_back({ 0.f, 0.f });
		pts.push_back({ 0.f, (2 <= tag_id && tag_id <= 7) ? (725.f) : (660.f) });
		pts.push_back({ (2 <= tag_id && tag_id <= 7) ? (780.f) : (1180.f), (2 <= tag_id && tag_id <= 7) ? (725.f) : (660.f) });
		pts.push_back({ (2 <= tag_id && tag_id <= 7) ? (780.f) : (1180.f), 0.f });
		return pts;
	}
};

class Detector {
    static constexpr int TOPK_NUM = 128;
    static constexpr float KEEP_THRES = 0.1f;
public:
	Detector();

	std::vector<box_t> process_frame(cv::Mat& img);

   bool init(std::string xml_path);

    bool uninit();



private:

    bool parse_yolov5(const Blob::Ptr &blob,std::vector<box_t> &boxes, float scale);
    
    //存储初始化获得的可执行网络
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    std::string _input_name;
 
    //参数区
    std::string  _xml_path;                             //OpenVINO模型xml文件路径



};
#endif
