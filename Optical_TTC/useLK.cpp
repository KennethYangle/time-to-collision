#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <math.h>
using namespace std; 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>


const double PI = 3.1415926;
void drawArrow(cv::Mat& img, cv::Point2f pStart, cv::Point2f pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType)
{
    cv::Point2f arrow;
    //计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）
    double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
    line(img, pStart, pEnd, color, thickness, lineType);
    //计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置）
    arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
}

int main( int argc, char** argv )
{
    // if ( argc != 2 )
    // {
    //     cout<<"usage: useLK path_to_dataset"<<endl;
    //     return 1;
    // }
    // string path_to_dataset = argv[1];
    // string associate_file = path_to_dataset + "/associate.txt";
    
    // ifstream fin( associate_file );
    // if ( !fin ) 
    // {
    //     cerr<<"I cann't find associate.txt!"<<endl;
    //     return 1;
    // }
    
    // string rgb_file, depth_file, time_rgb, time_depth;
    list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, depth, last_color;
    // cv::VideoCapture cap("/home/kenneth/Workspace/OpenCv/Optical_TTC/data/c280a7a1ff65d4567f5ba2a7ef1072f9.mp4");
    cv::VideoCapture cap(0);
    string output_video_path = "../capture.avi";
    cv::Size sWH = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter output_video;
    output_video.open(output_video_path, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);
    
    for ( int index=0; index<1000; index++ )
    {
        // fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        // color = cv::imread( path_to_dataset+"/"+rgb_file );
        // depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );

        cap >> color;

        if (index % 50 == 0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( color, kps );
            for ( auto kp:kps )
                keypoints.push_back( kp.pt );
            last_color = color.clone();
            continue;
        }
        // if ( color.data==nullptr || depth.data==nullptr )
        //     continue;
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints;
        vector<cv::Point2f> prev_keypoints;
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error; 
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉
        int i=0;
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
        {
            if ( status[i] == 0 )
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }
        i = 0;
        for ( auto iter=prev_keypoints.begin(); iter!=prev_keypoints.end(); i++)
        {
            if ( status[i] == 0 )
            {
                iter = prev_keypoints.erase(iter);
                continue;
            }
            iter++;
        }

        cout<<"tracked keypoints: "<<keypoints.size()<<endl;
        cout<<"prev_keypoints_size: "<<prev_keypoints.size()<<endl;
        cout<<"next_keypoints_size: "<<next_keypoints.size()<<endl;
        if (keypoints.size() == 0)
        {
            cout<<"all keypoints are lost."<<endl;
            break;
        }
        // 画出 keypoints
        cv::Mat img_show = color.clone();
        // for ( auto kp:prev_keypoints )
        //     cv::circle(img_show, kp, 3, cv::Scalar(0, 0, 240), 1);
        // for ( auto kp:keypoints )
        //     cv::circle(img_show, kp, 3, cv::Scalar(0, 240, 0), 1);
        auto iter_prev = prev_keypoints.begin();
        auto iter_next = keypoints.begin();
        cv::Scalar line_color;
        int theta_to_scale = int( atan( abs( (iter_next->y-iter_prev->y) / (iter_next->x-iter_prev->x) ) /PI*2*255 ) );
        if ( iter_prev->x > iter_next->x && iter_prev->y > iter_next->y )
            line_color = cv::Scalar(10, theta_to_scale, 10);
        else if ( iter_prev->x > iter_next->x && iter_prev->y < iter_next->y )
            line_color = cv::Scalar(60, 60, theta_to_scale);
        else if ( iter_prev->x < iter_next->x && iter_prev->y > iter_next->y )
            line_color = cv::Scalar(theta_to_scale, 10, 110);
        else
            line_color = cv::Scalar(theta_to_scale, 160, 160);
        

        for ( ; iter_next!=keypoints.end(); )
        {
            drawArrow(img_show, *iter_prev, *iter_next, 2, 15, line_color, 1, 8);
            iter_prev++;
            iter_next++;
        }

        cv::imshow("corners", img_show);
        cv::waitKey(1);
        output_video << img_show;
        last_color = color.clone();
    }
    output_video.release();
    return 0;
}
