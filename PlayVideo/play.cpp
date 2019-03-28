#include<opencv2/opencv.hpp>
using namespace cv;

int main(){
    VideoCapture cap;
    cap.open("/home/kenneth/Workspace/OpenCv/PlayVideo/VID_20190328_214554.mp4"); //打开视频,以上两句等价于VideoCapture cap("E://2.avi");

    //cap.open("http://www.laganiere.name/bike.avi");//也可以直接从网页中获取图片，前提是网页有视频，以及网速够快
    if(!cap.isOpened())//如果视频不能正常打开则返回
        return -1;
    Mat frame;
    while(1)
    {
        cap>>frame;//等价于cap.read(frame);
        if(frame.empty())//如果某帧为空则退出循环
            break;
        imshow("video", frame);
        waitKey(1);//每帧延时20毫秒
    }
    cap.release();//释放资源
    return 0;
}