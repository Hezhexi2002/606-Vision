#include "sjtu.h"



template<class F, class T, class ...Ts>
T reduce(F &&func, T x, Ts ...xs) {
	if constexpr (sizeof...(Ts) > 0) {
		return func(x, reduce(std::forward<F>(func), xs...));
	}
	else {
		return x;
	}
}

template<class T, class ...Ts>
T reduce_min(T x, Ts ...xs) {
	return reduce([](auto a, auto b) { return std::min(a, b); }, x, xs...);
}

template<class T, class ...Ts>
T reduce_max(T x, Ts ...xs) {
	return reduce([](auto a, auto b) { return std::max(a, b); }, x, xs...);
}

// 判断目标外接矩形是否相交，用于nms。
// 等效于thres=0的nms。
static inline bool is_overlap(const cv::Point2f pts1[4], const cv::Point2f pts2[4]) {
	cv::Rect2f box1, box2;
	box1.x = reduce_min(pts1[0].x, pts1[1].x, pts1[2].x, pts1[3].x);
	box1.y = reduce_min(pts1[0].y, pts1[1].y, pts1[2].y, pts1[3].y);
	box1.width = reduce_max(pts1[0].x, pts1[1].x, pts1[2].x, pts1[3].x) - box1.x;
	box1.height = reduce_max(pts1[0].y, pts1[1].y, pts1[2].y, pts1[3].y) - box1.y;
	box2.x = reduce_min(pts2[0].x, pts2[1].x, pts2[2].x, pts2[3].x);
	box2.y = reduce_min(pts2[0].y, pts2[1].y, pts2[2].y, pts2[3].y);
	box2.width = reduce_max(pts2[0].x, pts2[1].x, pts2[2].x, pts2[3].x) - box2.x;
	box2.height = reduce_max(pts2[0].y, pts2[1].y, pts2[2].y, pts2[3].y) - box2.y;
	return (box1 & box2).area() > 0;
}

static inline int argmax(const float *ptr, int len) {
	int max_arg = 0;
	for (int i = 1; i < len; i++) {
		if (ptr[i] > ptr[max_arg]) max_arg = i;
	}
	return max_arg;
}

float inv_sigmoid(float x) {
	return -std::log(1 / x - 1);
}

float sigmoid(float x) {
	return 1 / (1 + std::exp(-x));
}

bool Detector::init(string xml_path){
    _xml_path = xml_path;
    
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 
    //输入设置
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    //输出设置
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //获取可执行网络
    //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}

//释放资源
bool Detector::uninit(){
    return true;
}

bool Detector::parse_yolov5(const Blob::Ptr &blob, std::vector<box_t> &boxes, float scale){
    
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();
    
        // 模型后处理
    std::vector<box_t> before_nms = boxes;
    for (int i = 0; i < 640; i++)
    {
        float *result = (float *)output_blob + i * 640;
        if (result[8] < inv_sigmoid(0.5))
            continue;
        box_t box;
        for (int i = 0; i < 4; i++)
        {
            box.pts[i].x = (result[i * 2 + 0]) / scale;
            box.pts[i].y = (result[i * 2 + 1]) / scale;
        }
        box.color_id = argmax(result + 9, 4);
        box.tag_id = argmax(result + 13, 9);
        box.conf = sigmoid(result[8]);
        before_nms.push_back(box);
    }
    std::sort(before_nms.begin(), before_nms.end(), [](box_t &b1, box_t &b2)
              { return b1.conf > b2.conf; });
    boxes.clear();
    boxes.reserve(before_nms.size());
    std::vector<bool> is_removed(before_nms.size());
    for (int i = 0; i < before_nms.size(); i++)
    {
        if (is_removed[i])
            continue;
        boxes.push_back(before_nms[i]);
        for (int j = i + 1; j < before_nms.size(); j++)
        {
            if (is_removed[j])
                continue;
            if (is_overlap(before_nms[i].pts, before_nms[j].pts))
                is_removed[j] = true;
        }
    }
    if(before_nms.size() == 0) return false;
    else return true;
}


std::vector<box_t> Detector::process_frame(cv::Mat& img) {
    
    float scale = 640.f / std::max(img.cols, img.rows);
    cv::resize(img, img, {(int)round(img.cols * scale), (int)round(img.rows * scale)});
    cv::Mat input(640, 640, CV_8UC3, 127);
    img.copyTo(input({0, 0, img.cols, img.rows}));
    cvtColor(input, input, COLOR_BGR2RGB);
    // cv::Mat x;

    size_t img_size = 640*640;
    InferRequest infer_request = _network.CreateInferRequest();
    Blob::Ptr frameBlob = infer_request.GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    
    //nchw
    for(size_t row =0;row<640;row++){
        for(size_t col=0;col<640;col++){
            for(size_t ch =0;ch<3;ch++){
                blob_data[img_size*ch + row*640 + col] = float(input.at<Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }
    //执行预测
    infer_request.Infer();
    

    // 模型后处理
    std::vector<box_t> rst;

    int i = 0;
for(auto &output : _outputinfo){
    auto output_name = output.first;
    Blob::Ptr blob = infer_request.GetBlob(output_name);
    parse_yolov5(blob, rst, scale);
  
    ++i;
}
    return rst;
}