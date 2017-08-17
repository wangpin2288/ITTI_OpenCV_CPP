#ifndef ITTI_H
#define ITTI_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
#define PI 3.1415926  

class ITTI
{
public:
	ITTI(Mat src);
	Mat getSalMap();
	Mat Normanization(Mat src);//归一化N操作
	void OSplit(Mat src);//gabor滤波
	Mat splitCFeature(Mat src);
	Mat splitIFeature(Mat src);
	Mat splitOFeature(Mat src);

	Mat downSample(Mat src,int x);//下采样
	Mat upSample(Mat src,int x);//上采样
	Mat overScaleSub(Mat src,int c,int s);//根据输入图像和c,s尺寸参数，得到跨尺度相减后的图像
	void localMaxima(Mat& src,const int minPeakDistance,
					vector<double>& localMax,
					vector<double>& localMin);//求图像的局部最大值

	Mat getMyGabor(int width, int height, 
				   int U, int V, double Kmax, double f,
   				   double sigma, int ktype, 
				   const string& kernel_name);

	void construct_gabor_bank();

//分离bgr,RGBY,I,O通道分量
	Mat bSplit(Mat src);
	Mat gSplit(Mat src);
	Mat rSplit(Mat src);
	Mat RSplit(Mat src);
	Mat GSplit(Mat src);
	Mat BSplit(Mat src);
	Mat YSplit(Mat src);
	Mat ISplit(Mat src);

private:
	Mat srcImg;
	vector<Mat&> gaborMatVec;//存储图像4个方向特征
	int c[3];
	int delta[2];
};




















#endif
