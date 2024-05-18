#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159
#endif
using namespace cv;
using namespace std;
int findMaxIndex(double arr[], int size);
std::vector<cv::Point> findPoint(Point match, double angle, Mat img);
double distance(Point A,Point B, Mat img1);
int main()
{
    double maxMatch[21]={0};
    Point maxmatchPosition[21];
    Mat image = imread("D:\\Project_Code\\C_CPP\\test1Clion\\pic\\model.png");
    Mat grayImg,binaryImg,maskImg;
    cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);
    threshold(grayImg, binaryImg, 30, 255, cv::THRESH_BINARY);
    bitwise_not(binaryImg, maskImg);
    waitKey(0);
    string file_template = "D:\\Project_Code\\C_CPP\\test1Clion\\pic\\spark_plug_{}.png";
    string file_output_template = "D:\\Project_Code\\C_CPP\\test1Clion\\pic1\\spark_plug_{}.png";
    int num_pic=20;
    for(int i=1;i <= num_pic; i++)
    {
        string file_name = file_template;
        size_t pos = file_name.find("{}");
        file_name.replace(pos, 2, std::to_string(i));
        Mat imageProcess = imread(file_name);
        for(float j=0;j<=5;j=j+0.25)
        {
            Mat result;
            Mat rotateTemp,rotateMask;
            Point2f pt = Point2f((float)image.cols / 2, (float)image.rows / 2);
            Mat M = getRotationMatrix2D(pt, j, 1);
            warpAffine(image, rotateTemp, M, image.size());
            warpAffine(maskImg, rotateMask, M, image.size());
            matchTemplate(imageProcess, rotateTemp, result, TM_CCOEFF_NORMED, rotateMask);
            double minValue;
            double maxValue;
            Point minLocation;
            Point maxLocation;
            Point matchLocation;
            minMaxLoc(result, &minValue, &maxValue, &minLocation, &maxLocation, Mat());
            matchLocation = maxLocation;
            maxMatch[int(4*j)]=maxValue;
            maxmatchPosition[int(4*j)]=matchLocation;
        }
        int size = sizeof(maxMatch) / sizeof(maxMatch[0]);
        int maxIndex = findMaxIndex(maxMatch, size);
        printf("%f\n",0.25*maxIndex);
        std::vector<cv::Point> RECpoints;
        RECpoints=findPoint(maxmatchPosition[maxIndex],0.25*maxIndex,image);
        double dis=distance(RECpoints[3],RECpoints[2],imageProcess);
        printf("%f\n",dis);
        std::string numberString = std::to_string(dis);
        cv::putText(imageProcess, numberString, cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        string file_output_name = file_output_template;
        const cv::Point* ppt[1] = {&RECpoints[0]};
        int npt[] = {4};
        cv::polylines(imageProcess, ppt, npt, 1, true, cv::Scalar(0, 0, 255));
        size_t pos1 = file_output_name.find("{}");
        file_output_name.replace(pos1, 2, std::to_string(i));
        imwrite(file_output_name, imageProcess);
    }
    return 0;
}
int findMaxIndex(double arr[], int size)
{
    double* maxPtr = std::max_element(arr, arr + size);
    int maxIndex = std::distance(arr, maxPtr);
    return maxIndex;
}
std::vector<cv::Point> findPoint(Point match, double angle, Mat img)
{
    double radian = -angle * CV_PI / 180.0;
    cv::Point topRight(match.x + img.cols, match.y);
    cv::Point bottomLeft(match.x, match.y + img.rows);
    cv::Point bottomRight(match.x + img.cols, match.y + img.rows);
    std::vector<cv::Point> rotatedPoints(4);
    auto rotatePoint = [&](const cv::Point& p){
        double x = (p.x - match.x) * cos(radian) - (p.y - match.y) * sin(radian) + match.x;
        double y = (p.x - match.x) * sin(radian) + (p.y - match.y) * cos(radian) + match.y;
        return cv::Point(cvRound(x), cvRound(y));
    };
    rotatedPoints[0] = match;
    rotatedPoints[1] = rotatePoint(topRight);
    rotatedPoints[2] = rotatePoint(bottomRight);
    rotatedPoints[3] = rotatePoint(bottomLeft);

    return rotatedPoints;
}
double distance(Point A,Point B, Mat img1)
{
    cv::Point midPoint((A.x + B.x) / 2, (A.y + B.y) / 2);//找到中点
    double slope = -1.0 / ((double)(B.y - A.y) / (B.x - A.x));//计算斜率
    std::vector<int> grayValues;
    Mat grayImg1;
    cvtColor(img1, grayImg1, cv::COLOR_BGR2GRAY);
    vector<Point> LinePoint1;
    vector<Point> LinePoint;
    vector<int> IndexOfEdge;
    if(B.y - A.y==0)
    {
        for(int k=midPoint.y;k<=img1.rows;k++)
        {
            LinePoint.push_back(Point(midPoint.x,k));
        }
    }
    else
    {
        for (int x = midPoint.x; x <= std::max(A.x, B.x); x++)
        {
            vector<int> y1;
            int y;
            y = int(slope * (x - midPoint.x) + midPoint.y);
            if (y >= 0 && y < img1.rows)
            {
                y1.push_back(y);
                LinePoint1.push_back(Point(x, y));
            }
        }
        LinePoint1.push_back(Point(LinePoint1.back().x, img1.rows));
        cv::LineIterator it(img1, LinePoint1[0], LinePoint1.back(), 8);
        for (int i = 0; i < it.count; i++, ++it)
        {
            LinePoint.push_back(it.pos());
        }
    }
        for (size_t i = 0; i < LinePoint.size()-1; i++)
        {
            int grayValue = grayImg1.at<uchar>(LinePoint[i].y, LinePoint[i].x);
            grayValues.push_back(grayValue);
        }
        std::vector<int> differences;
        for (size_t i = 1; i < grayValues.size(); i++)
        {
            int diff = grayValues[i] - grayValues[i - 1];
            if (diff <= -50)
            {
                differences.push_back(diff);
                IndexOfEdge.push_back(i);
            }
        }
        vector<int> Edgeplus;
    std::vector<int> differences1;
    for (size_t i = 1; i < grayValues.size(); i++)
    {
        int diff = grayValues[i] - grayValues[i - 1];
        if (diff >= +50)
        {
            differences1.push_back(diff);
            Edgeplus.push_back(i);
        }
    }
    cv::line(img1, A, B, cv::Scalar(255), 2);
    cv::line(img1, LinePoint[Edgeplus[0]], LinePoint.back(), cv::Scalar(255), 1);
    cv::circle(img1, midPoint, 3, cv::Scalar(255), -1);
    double distanceToEdge = cv::norm(LinePoint[Edgeplus[0]] - LinePoint[IndexOfEdge.back()]);
    return distanceToEdge;
}