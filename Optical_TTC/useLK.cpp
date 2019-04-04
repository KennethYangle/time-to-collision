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
    list< cv::Point2f > keypoints;                  // 因为要删除跟踪失败的点，使用list
    cv::Mat color, last_color;
    cv::VideoCapture cap(0);                        // 摄像头捕捉或载入视频
    // cv::VideoCapture cap("/home/kenneth/Workspace/OpenCv/Optical_TTC/data/c280a7a1ff65d4567f5ba2a7ef1072f9.mp4");
    string output_video_path = "../capture.avi";    // 保存视频
    cv::Size sWH = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter output_video;
    output_video.open(output_video_path, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);
    
    for ( int index=0; index<1000; index++ )
    {
        cap >> color;

        // 每50帧提取FAST特征点
        if (index % 50 == 0 )
        {
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( color, kps );
            for ( auto kp:kps )
                keypoints.push_back( kp.pt );
            last_color = color.clone();
            continue;
        }
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints;
        vector<cv::Point2f> prev_keypoints;
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);   // 前一帧的特征点变成这一帧的prev
        vector<unsigned char> status;
        vector<float> error;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

        // 把跟丢的点删掉
        int i = 0;
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
        if (keypoints.size() == 0)
        {
            cout<<"all keypoints are lost."<<endl;
            break;
        }

        // 计算光流
        vector<cv::Point2f> optical_flow;
        i = 0;
        cv::Point2f temp;
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
        {
            temp.x = iter->x - prev_keypoints[i].x;
            temp.y = iter->y - prev_keypoints[i].y;
            optical_flow.push_back(temp);
            iter++;
        }

        // 计算延伸焦点 Focus of Extend
        cv::Mat A(optical_flow.size(), 2, CV_64F);
        cv::Mat b(optical_flow.size(), 1, CV_64F);
        cv::Mat foe(2, 1, CV_64F);
        cv::Mat optical_translation(optical_flow.size(), 2, CV_64F);
        for ( int i=0; i<optical_flow.size(); i++)  // 计算光流平移分量
        {
            auto pxvec = optical_translation.ptr<double>(i);
            pxvec[0] = optical_flow[i].x;           // 应该减去旋转分量
            pxvec[1] = optical_flow[i].y;
        }
        auto iter_key = keypoints.begin();
        for ( int i=0; i<A.rows; i++)               // 计算A，b
        {
            auto px_A = A.ptr<double>(i);
            auto px_b = b.ptr<double>(i);
            auto px_trans = optical_translation.ptr<double>(i);
            px_A[0] = px_trans[1];
            px_A[1] = -px_trans[0];
            px_b[0] = iter_key->x * px_trans[1] - iter_key->y * px_trans[0];
            iter_key++;
        }
        cv::solve(A, b, foe, cv::DECOMP_QR);        // 求解FOE
        cout << "FOE: \n" << foe << endl;

        // 画出 keypoints和光流
        cv::Mat img_show = color.clone();
        cv::Scalar line_color;
        cv::circle(img_show, cv::Point(int(foe.at<double>(0,0)), int(foe.at<double>(1,0))), 5, cv::Scalar(0, 0, 240), 3);     // 画FOE
        // for ( auto kp:prev_keypoints )
        //     cv::circle(img_show, kp, 3, cv::Scalar(0, 0, 240), 1);
        // for ( auto kp:keypoints )
        //     cv::circle(img_show, kp, 3, cv::Scalar(0, 240, 0), 1);
        i = 0;
        for ( auto iter=optical_flow.begin(); iter!=optical_flow.end(); iter++)
        {
            int theta_to_scale = int( atan( abs(iter->y/iter->x) ) /PI*2*255 );
            if ( iter->x > 0 && iter->y >0)
                line_color = cv::Scalar(10, theta_to_scale, 10);
            else if ( iter->x > 0 && iter->y < 0)
                line_color = cv::Scalar(60, 60, theta_to_scale);
            else if ( iter->x < 0 && iter->y > 0)
                line_color = cv::Scalar(theta_to_scale, 10, 110);
            else
                line_color = cv::Scalar(theta_to_scale, 160, 160);
            drawArrow(img_show, prev_keypoints[i], prev_keypoints[i]+*iter, 2, 15, line_color, 1, 8);
            i++;
        }

        cv::imshow("corners", img_show);
        cv::waitKey(1);
        output_video << img_show;
        last_color = color.clone();
    }
    output_video.release();
    return 0;
}
