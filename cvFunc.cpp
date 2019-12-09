#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "cvFunc.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <string.h>
using namespace std;
using namespace cv; //包含cv命名空间


void ShowVideo(char *file){
    VideoCapture capture(file);
    if(!capture.isOpened()){
        cout<<"读取失败"<<endl;
        return;
    }
    Mat frame,res;
    double rate=capture.get(CV_CAP_PROP_FPS);
    int delay=1000/rate;
    namedWindow("Video");
    while(1){
        if(!capture.read(frame))
            break;
        else{
            //cvtColor(frame,frame,COLOR_RGB2GRAY);
            //Canny(frame,frame,50,200,3);
            PlateDetect(frame,res);
            imshow("Video",res);
            waitKey(delay);
        }
    }
}

void KmeansSeg(char *file){
    Mat srcImage=imread(file);
    if(srcImage.empty()){
        cout<<"could not load image...\n";
        return;
    }
    imshow("Original",srcImage);
    Scalar colorTab[5]={Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0),Scalar(0,255,255),Scalar(255,0,255)};
    int width=srcImage.cols;
    int height=srcImage.rows;
    int channels=srcImage.channels();
    int sampleCount=width*height;
    int clusterCount=5;

    Mat points(sampleCount,channels,CV_32F,Scalar(10));
    Mat labels;
    Mat center(clusterCount,1,points.type());
    int index;
    for(int i=0;i<srcImage.rows;i++){
        for(int j=0;j<srcImage.cols;j++){
            index=i*width+j;
            Vec3b bgr=srcImage.at<Vec3b>(i,j);
            points.at<float>(index,0)=static_cast<int>(bgr[0]);
            points.at<float>(index,1)=static_cast<int>(bgr[1]);
            points.at<float>(index,2)=static_cast<int>(bgr[2]);
        }
    }
    TermCriteria criteria=TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,0.1);
    kmeans(points,clusterCount,labels,criteria,3,KMEANS_PP_CENTERS,center);
    Mat result=Mat::zeros(srcImage.size(),srcImage.type());
    for(int i=0;i<srcImage.rows;i++){
        for(int j=0;j<srcImage.cols;j++){
            index=i*width+j;
            int label=labels.at<int>(index);
            result.at<Vec3b>(i,j)[0]=colorTab[label][0];
            result.at<Vec3b>(i,j)[1]=colorTab[label][1];
            result.at<Vec3b>(i,j)[2]=colorTab[label][2];
        }
    }
    imshow("Kmeans",result);
    waitKey(0);
}

double DegreeTrans(double theta){
    return (theta/CV_PI)*180;
}
//将图像旋转degree角度
void rotateImage(Mat src,Mat &img_rotate,double degree){
    Point2f center;
    center.x=float(src.cols/2.0);
    center.y=float(src.rows/2.0);
    int length=sqrt(src.rows*src.rows+src.cols*src.cols);
    Mat M=getRotationMatrix2D(center,degree,1);
    warpAffine(src,img_rotate,M,Size(length,length),1,0,Scalar(255,255,255));
}
//使用霍夫变换检测直线，进而计算倾斜角度
double CalcDegree(const Mat& srcImage){
    Mat midImage,dstImage;
    Canny(srcImage,midImage,50,200,3);
    cvtColor(midImage,dstImage,CV_GRAY2BGR);
    vector<Vec2f> lines;
    HoughLines(midImage,lines,1,CV_PI/180,300,0,0);
    if(!lines.size())
        HoughLines(midImage,lines,1,CV_PI/180,200,0,0);
    if(!lines.size())
        HoughLines(midImage,lines,1,CV_PI/180,150,0,0);
    if(!lines.size()){
        cout<<"没有检测到直线"<<endl;
        return -1;
    }
    float sum=0;
    for(int i=0;i<lines.size();i++){
        float rho=lines[i][0];
        float theta=lines[i][1];
        Point pt1,pt2;
        double a=cos(theta),b=sin(theta);
        double x0=a*rho,y0=b*rho;
        pt1.x=cvRound(x0+1000*(-b));
        pt1.y=cvRound(y0+1000*(a));
        pt2.x=cvRound(x0-1000*(-b));
        pt2.y=cvRound(y0-1000*(a));
        sum+=theta;
        line(dstImage,pt1,pt2,Scalar(55,100,195),1,LINE_AA);

    }
    //imshow("检测效果图",dstImage);
    float average=sum/(lines.size());
    double angle=DegreeTrans(average)-90;
    return angle;
}
//对图像进行矫正
void RectifyImage(char *file){
    double degree;
    Mat src=imread(file);
    imshow("原始图",src);
    Mat dst;
    degree=CalcDegree(src);
    if(degree==-1){
        cout<<"矫正失败"<<endl;
        return;
    }
    rotateImage(src,dst,degree);
    imshow("矫正后",dst);
    waitKey(0);
}
















//基于颜色特征对车牌进行定位
void PlateDetect(Mat OriginalImg,Mat &resImage){
    //Mat OriginalImg;
	//OriginalImg=imread(file);
	if(OriginalImg.empty()){
        cout<<"错误！读取图像失败\n";
        return;
	}
	Mat ResizeImg=OriginalImg.clone();
	//if(OriginalImg.cols>640)
    resize(OriginalImg,ResizeImg,Size(640,640*OriginalImg.rows/OriginalImg.cols));
    unsigned char pixelB,pixelG,pixelR;
    unsigned char DifMax=50;
    unsigned char B=138,G=63,R=23;
    Mat BinRGBImg=ResizeImg.clone();
    for(int i=0;i<ResizeImg.rows;i++)
    for(int j=0;j<ResizeImg.cols;j++){
        pixelB=ResizeImg.at<Vec3b>(i,j)[0];
        pixelG=ResizeImg.at<Vec3b>(i,j)[1];
        pixelR=ResizeImg.at<Vec3b>(i,j)[2];
        if(abs(pixelB-B)<DifMax && abs(pixelG-G)<DifMax && abs(pixelR-R)<DifMax){
            BinRGBImg.at<Vec3b>(i,j)[0]=255;
            BinRGBImg.at<Vec3b>(i,j)[1]=255;
            BinRGBImg.at<Vec3b>(i,j)[2]=255;
        }
        else{
            BinRGBImg.at<Vec3b>(i,j)[0]=0;
            BinRGBImg.at<Vec3b>(i,j)[1]=0;
            BinRGBImg.at<Vec3b>(i,j)[2]=0;
        }
    }
    Mat BinOriImg;
    Mat element=getStructuringElement(MORPH_RECT,Size(4,4));
    Mat element1=getStructuringElement(MORPH_RECT,Size(1,2));
    erode(BinRGBImg,BinOriImg,element1);
    dilate(BinOriImg,BinOriImg,element);
    dilate(BinOriImg,BinOriImg,element);
    dilate(BinOriImg,BinOriImg,element);
    erode(BinOriImg,BinOriImg,element1);
    erode(BinOriImg,BinOriImg,element1);
    dilate(BinOriImg,BinOriImg,element);
    dilate(BinOriImg,BinOriImg,element);
    dilate(BinOriImg,BinOriImg,element);

    //imshow("ss",BinOriImg);
    double length,area,rectArea;
    double rectDegree=0.0;
    double long2short=0.0;
    CvRect rect;
    CvBox2D box,boxTemp;
    CvPoint2D32f pt[4];
    double axisLong=0.0,axisShort=0.0;
    double axisLongTemp=0.0,axisShortTemp=0.0;
    double LengthTemp;
    float angle=0;
    float angleTemp=0;
    bool TestPlantFlag=0;

    cvtColor(BinOriImg,BinOriImg,CV_BGR2GRAY);
    threshold(BinOriImg,BinOriImg,100,255,THRESH_BINARY);
    CvMemStorage *storage=cvCreateMemStorage(0);
    CvSeq *seq=0;
    CvSeq *tempSeq=cvCreateSeq(CV_SEQ_ELTYPE_POINT,sizeof(CvSeq),sizeof(CvPoint),storage);
    IplImage temp=IplImage(BinOriImg);
    IplImage *qImg=&temp;
    Mat GuiRGBImg=ResizeImg.clone();
    int cnt = cvFindContours(qImg, storage, &seq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    //cout<<"轮廓数为："<<cnt<<endl;
    for(tempSeq=seq;tempSeq!=NULL;tempSeq=tempSeq->h_next){
            length=cvArcLength(tempSeq);
            area=cvContourArea(tempSeq);
            if(area>1500){
                rect=cvBoundingRect(tempSeq,1);
                boxTemp=cvMinAreaRect2(tempSeq,0);
                cvBoxPoints(boxTemp,pt);
                angleTemp=boxTemp.angle;

                axisLongTemp=sqrt(pow(pt[1].x-pt[0].x,2)+pow(pt[1].y-pt[0].y,2));
                axisShortTemp=sqrt(pow(pt[2].x-pt[1].x,2)+pow(pt[2].y-pt[1].y,2));
              //waitKey(0);
                if(axisShortTemp>axisLongTemp){
                    LengthTemp=axisLongTemp;
                    axisLongTemp=axisShortTemp;
                    axisShortTemp=LengthTemp;
                }
                else
                    angleTemp+=90;
                rectArea=axisLongTemp*axisShortTemp;
                rectDegree=rectArea/area;
                long2short=axisLongTemp/axisShortTemp;
                //cout<<long2short<<" "<<rectDegree<<" "<<rectArea<<endl;
                if(long2short>1.5 && long2short<5 && rectDegree>0.63 && rectDegree<3 && rectArea>2000 && rectArea<20000){
                    TestPlantFlag=true;
                    for(int i=0;i<4;i++){
                        temp=IplImage(GuiRGBImg);
                        qImg=&temp;
                        cvLine(qImg,cvPointFrom32f(pt[i]),cvPointFrom32f(pt[(i+1)%4?(i+1):0]),CV_RGB(255,0,0));
                    }
                    box=boxTemp;
                    angle=angleTemp;
                    axisLong=axisLongTemp;
                    axisShort=axisShortTemp;
                }
                else if(rectArea>2000 && rectArea<8000){
                    for(int i=0;i<4;i++){
                        temp=IplImage(GuiRGBImg);
                        qImg=&temp;
                        cvLine(qImg,cvPointFrom32f(pt[i]),cvPointFrom32f(pt[(i+1)%4?(i+1):0]),CV_RGB(0,255,0));
                    }
                }
            }
    }
    resImage=GuiRGBImg;
    //imshow("Result",GuiRGBImg);
	//waitKey(0); //等待任意按键按下
}
