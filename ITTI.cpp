#include "ITTI.h"  

ITTI::ITTI(Mat src)
{
	srcImg = src;
	c[0]=2;
	c[1]=3;
	c[2]=4;
	delta[0]=3;
	delta[1]=4;
}


Mat ITTI::getSalMap()
{
	Mat src = srcImg;
	Mat cNFeature = Normanization( splitCFeature(src) );
	Mat iNFeature = Normanization( splitIFeature(src) );
	Mat oNFeature = Normanization( splitOFeature(src) );
	
	int height = cNFeature.rows;
	int width = cNFeature.cols;

	Mat salMap = cNFeature;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			salMat.at<uchar>(i,j) = (cNFeature.at<uchar>(i,j) + iNFeature.at<uchar>(i,j) + oNFeature.at<uchar>(i,j))/3;
		}
	}

	return salMap;
}

//N操作(对单通道图像进行归一化操作)
Mat ITTI::Normanization(Mat src)
{
	Mat dst(src.rows, src.cols, CV_64FC1);
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			dst.at<double>(i, j) = (double)img.at<uchar>(i, j);
		}
	}

   	vector<double> localMax;
  	vector<double> localMin;
  	double max=0,min=255,s,m;

  	localMaxima(dst,2,localMax,localMin);

   	for(int i=0;i<localMax.size();i++)
   	{
     	 if(localMax[i]>max) max=localMax[i];
     	 if(localMin[i]<min) min=localMax[i];
     	 s+=localMax[i];
   	}

    for(int i=0;i<height;i++)
     for(int j=0;j<width;j++)
     {
         double val=(dst.at<double>(i,j)-min)/(double)(max-min)*100;
         dst.at<double>(i,j)=val;
     }

    s = (s-localMax.size()*min)/(double)(max-min)*100;
    max = 100;
    min = 0;
    m = (s-max)/(double)(localMax.size()-1);

    double d=(max-m)*(max-m);
  
    for(int i=0;i<height;i++)
	for(int j=0;j<width;j++)
	{
	   dst.at<double>(i,j) = dst.at<double>(i,j)*d;
	}

    Mat dst1(height, width, CV_8UC1);
    double maxDst = 100*d;
    double minDst = 0;
 //归一化后的像素值还原为0-255之间的整数
	for(int i=0;i<height;i++)
    for(int j=0;j<width;j++)
    {
        double v = dst.at<double>(i,j)/maxDst*255;
        dst1.at<uchar>(i,j)=(int)v;
    }
	return dst1;
}

//Gabor滤波，分离方向向量
void ITTI::OSplit(Mat img)
{
	// variables for gabor filter
	double Kmax = PI/2;
	double f = sqrt(2.0);
	double sigma = 2*PI;
	int U = 7;
	int V = 4;
	int GaborH = 129;
	int GaborW = 129;

	Mat kernel_re, kernel_im;
	Mat dst_re, dst_im, dst_mag;

	// variables for filter2D
	Point archor(-1,-1);
	int ddepth = -1;
	double delta = 0;

	// filter image with gabor bank
	Mat totalMat;
	for(U = 0; U < 8; U+=2){
		Mat colMat;
		for(V = 0;V < 2; V+=2){
			kernel_re = getMyGabor(GaborW, GaborH, U, V,
			Kmax, f, sigma, CV_64F, "real");
			kernel_im = getMyGabor(GaborW, GaborH, U, V,
			Kmax, f, sigma, CV_64F, "imag");

				filter2D(img, dst_re, ddepth, kernel_re);
				filter2D(img, dst_im, ddepth, kernel_im);

				dst_mag.create(img.rows, img.cols, CV_32FC1);
				magnitude(Mat_<float>(dst_re),Mat_<float>(dst_im), 
					dst_mag);

				//show gabor kernel
				normalize(dst_mag, dst_mag, 0, 1, CV_MINMAX);
				printf("U%dV%d\n", U, V);
				imshow("dst_mag", dst_mag);
                        normalize(dst_mag,dst_mag,0,255,CV_MINMAX);
                        dst_mag.convertTo(dst_mag,CV_8U);
                        gaborMatVec.puch_back(dst_mag);
			}
		}
}


Mat ITTI::splitIFeature(Mat src)
{
	Mat Img = ISplit(src);
	Mat Ics[3][2];
	for(int i=0;i<3;i++)
	for(int j=0;j<2;j++)
	{
		Ics[i][j] = overScaleSub(Img,Img,c[i],c[i]+delta[j]);	
		Ics[i][j] = downSample(Ics[i][j], 4-c[i]);	//统一到尺寸4
		Ics[i][j] = Normanization(Ics[i][j]);
	}

	Mat dst = Ics[0][0];
	for(int i=0;i<dst.rows;i++)
		for(int j=0;j<dst.cols;j++)
			dst.at<uchar>(i,j) = (Ics[0][0].at<uchar>(i,j)+
			Ics[0][1].at<uchar>(i,j)+Ics[1][0].at<uchar>(i,j)+
			Ics[1][1].at<uchar>(i,j)+Ics[2][0].at<uchar>(i,j)+
			Ics[2][1].at<uchar>(i,j))/6;

	return dst;
}

Mat ITTI::splitOFeature(Mat src)
{
    OSplit(src);
	Mat Ocs[4][3][2];

	for(int k=0;k<4;k++)
	{
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<2;j++)
			{
				Mat img = gaborMatVec[k];
				img = overScalSub(img,c[i],c[i]+delta[j]);
				img = downSample(img,4-c[i]);	
        		Ocs[k][i][j] = Normanization(img);			
			}
		}
	}

	Mat dst = Ocs[0][0][0];
	for(int x=0;x<dst.rows;x++)
	{
		for(int y=0;y<dst.cols;y++)
		{
			int val=0;
			for(int k=0;k<4;k++)
			{
				for(int i=0;i<3;i++)
				{
					for(int j=0;j<2;j++)
					{
						val += Ocs[k][i][j].at<uchar>(x,y);
					}
				}	 
			}
			dst.at<uchar>(x,y) = val/24;
		}
	}

	return dst;
}

Mat ITTI::splitCFeature(Mat src)
{
	Mat Rimg = RSplit(src);	
	Mat Gimg = GSplit(src);	
	Mat Bimg = BSplit(src);	
	Mat Yimg = YSplit(src);	
	
	Mat RGcs[3][2];
	Mat BYcs[3][2];

	for(int i=0;i<3;i++)
	{
		for(int j=0;j<2;j++)
		{
			int c = c[i],s = c[i]+delta[j];
			Mat rg_c = overScaleSub(Rimg,Gimg,c,c);
			Mat gr_s = overScaleSub(Gimg,Rimg,s,s);
			Mat img1 = overScaleSub(rg_c.gr_s,c,s);
			img1 = downSample(img1,4-c);
			RGcs[i][j] = Normanization(img1);

			Mat by_c = overScaleSub(Bimg,Yimg,c,c);
			Mat yb_s = overScaleSub(Yimg,Bimg,s,s);
			Mat img2 = overScaleSub(by_c,yb_s,c,s);
			img2 = downSample(img2,4-c);
			BYcs[i][j] = Normanization(img2);
		}
	}

	Mat dst = BYcs[0][0];
	for(int x=0;x<dst.rows;x++)
	{
		for(int y=0;y<dst.cols;y++)
		{
			int val=0;
			for(int i=0;i<3;i++)
			{
				for(int j=0;j<2;j++)
				{
					val+=RGcs[i][j].at<uchar>(x,y)+BYcs[i][j].at<uchar>(x,y);
				}
			}
			dst.at<uchar>(x,y) = val/12;
		}

	}
	return dst;	
}



Mat ITTI::downSample(Mat src,int x)
{
	Mat dst;
	while(x-- > 0)
	{
		pyrDown(src,dst);
		downSample(dst,x);
	}
	if(x == -1) return dst; 
}

	
Mat ITTI::upSample(Mat src,int x)
	{
		Mat dst;
		while(x-- > 0)
		{
			pyrUp(src,dst);
			upSample(dst,x);
		}
		if(x == -1) return dst;
	}

//跨尺度相减
Mat ITTI::overScaleSub(Mat src1,Mat src2,int c,int s)
{
	Mat cImg = downSample(src1,c);
	Mat sImg = downSample(src2,s);
    Mat scImg;
	resize(sImg,scImg,Size(cImg.rows,cImg.cols),0,0,CV_INTER_LINEAR);

	Mat dst(cImg);
	for(int i=0;i<cImg.rows;i++)
		for(int j=0;j<cImg.cols;j++)
			for(int k=0;k<3;k++)
				dst.at<Vec3b>(i,j)[k] = abs(cImg.at<Vec3b>(i,j)[k]-scImg.at<Vec3b>(i,j)[k]);
	return dst;
}


//求图像的局部最大值	
void ITTI::localMaxima(Mat& src,const int minPeakDistance,
					vector<double>& localMax,
					vector<double>& localMin)
{
	int kernelSize = minPeakDistance;
	int minLoc = 0;
	double minVal = 0,maxVal = 0;
	Point minLocPt,maxLocPt;	
	Mat m(src);

	for(int i=0;i<src.rows-kernelSize;i+=kernelSize)
	for(int j=0;j<src.cols-kernelSize;j+=kernelSize)
	{
		Mat m_part = m(Rect(i,j,kernelSize,kernelSize));
		minMatLoc(m_part,&minVal,&maxVal,&minLocPt,&maxLocPt);
		localMax.push_back(maxVal);
		localMin.push_back(minVal);
	}
}



Mat ITTI::getMyGabor(int width, int height, int U, int V, double Kmax, double f,
	double sigma, int ktype, const string& kernel_name)
{

	int half_width = width / 2;
	int half_height = height / 2;
	double Qu = PI*U/8;
	double sqsigma = sigma*sigma;
	double Kv = Kmax/pow(f,V);
	double postmean = exp(-sqsigma/2);

	Mat kernel_re(width, height, ktype);
	Mat kernel_im(width, height, ktype);
	Mat kernel_mag(width, height, ktype);

	double tmp1, tmp2, tmp3;
	for(int j = -half_height; j <= half_height; j++){
		for(int i = -half_width; i <= half_width; i++){
			tmp1 = exp(-(Kv*Kv*(j*j+i*i))/(2*sqsigma));
			tmp2 = cos(Kv*cos(Qu)*i + Kv*sin(Qu)*j) - postmean;
			tmp3 = sin(Kv*cos(Qu)*i + Kv*sin(Qu)*j);

			if(ktype == CV_32F)
				kernel_re.at<float>(j+half_height, i+half_width) = 
					(float)(Kv*Kv*tmp1*tmp2/sqsigma);
			else
				kernel_re.at<double>(j+half_height, i+half_width) = 
					(double)(Kv*Kv*tmp1*tmp2/sqsigma);

			if(ktype == CV_32F)
				kernel_im.at<float>(j+half_height, i+half_width) = 
					(float)(Kv*Kv*tmp1*tmp3/sqsigma);
			else
				kernel_im.at<double>(j+half_height, i+half_width) = 
					(double)(Kv*Kv*tmp1*tmp3/sqsigma);
		}
	}

	magnitude(kernel_re, kernel_im, kernel_mag);

	if(kernel_name.compare("real") == 0)
		return kernel_re;
	else if(kernel_name.compare("imag") == 0)
		return kernel_im;
	else if(kernel_name.compare("mag") == 0)
		return kernel_mag;
	else
		printf("Invalid kernel name!\n");
}

void ITTI::construct_gabor_bank()
{
	double Kmax = PI/2;
	double f = sqrt(2.0);
	double sigma = 2*PI;
	int U = 7;
	int V = 4;
	int GaborH = 129;
	int GaborW = 129;

	Mat kernel;
	for(U = 0; U < 4; U+=2)
	{
		Mat colMat;
		for(V = 0;V < 2; V+=2)
		{
			kernel = getMyGabor(GaborW, GaborH, U, V,
				Kmax, f, sigma, CV_64F, "real");

			//show gabor kernel
			normalize(kernel, kernel, 0, 1, CV_MINMAX);
			printf("U%dV%d\n", U, V);

			if(V == 0)
				colMat = kernel;
			else
				vconcat(colMat, kernel, colMat);
		}
	}
}




//分离bgr,RGBY,I,O通道分量
Mat ITTI::bSplit(Mat src)
{
	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC1);
	for(int i=0;i<src.rows;i++)
		for(int j=0;j<src.cols;j++)
			dst.at<uchar>(i,j) = src.at<Vec3b>(i,j)[0];

	return dst;
}
	
Mat ITTI::gSplit(Mat src)
{
	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC1);
	for(int i=0;i<src.rows;i++)
		for(int j=0;j<src.cols;j++)
			dst.at<uchar>(i,j) = src.at<Vec3b>(i,j)[1];

	return dst;	
}


Mat ITTI::rSplit(Mat src)
{
	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC1);
	for(int i=0;i<src.rows;i++)
		for(int j=0;j<src.cols;j++)
			dst.at<uchar>(i,j) = src.at<Vec3b>(i,j)[2];

	return dst;
}
	
Mat ITTI::RSplit(Mat src)
{
	Mat bImg = bSplit(src);
	Mat gImg = gSplit(src);
	Mat rImg = rSplit(src);
	Mat dst = bImg;
	for(int i=0;i<src.rows;i++)
	for(int j=0;j<src.cols;j++)
		dst.at<uchar>(i,j) = rImg.at<uchar>(i,j)-(gImg.at<uchar>(i,j)+bImg.at<uchar>(i,j))/2;

	return dst;
}

Mat ITTI::GSplit(Mat src)
{
	Mat bImg = bSplit(src);
	Mat gImg = gSplit(src);
	Mat rImg = rSplit(src);
	Mat dst = bImg;
	for(int i=0;i<src.rows;i++)
	for(int j=0;j<src.cols;j++)
		dst.at<uchar>(i,j) = gImg.at<uchar>(i,j)-(rImg.at<uchar>(i,j)+bImg.at<uchar>(i,j))/2;

	return dst;
}



Mat ITTI::BSplit(Mat src)
{
	Mat bImg = bSplit(src);
	Mat gImg = gSplit(src);
	Mat rImg = rSplit(src);
	Mat dst = bImg;
	for(int i=0;i<src.rows;i++)
	for(int j=0;j<src.cols;j++)
		dst.at<uchar>(i,j) = bImg.at<uchar>(i,j)-(rImg.at<uchar>(i,j)+gImg.at<uchar>(i,j))/2;

	return dst;
}
	
Mat ITTI::YSplit(Mat src)
{
	Mat bImg = bSplit(src);
	Mat gImg = gSplit(src);
	Mat rImg = rSplit(src);
	Mat dst = bImg;
	for(int i=0;i<src.rows;i++)
	for(int j=0;j<src.cols;j++)
		dst.at<uchar>(i,j) = (rImg.at<uchar>(i,j)+gImg.at<uchar>(i,j))/2 - abs(rImg.at<uchar>(i,j)-gImg.at<uchar>(i,j))/2 - bImg.at<uchar>(i,j);

	return dst;
}
	
Mat ITTI::ISplit(Mat src)
{
	Mat bImg = bSplit(src);
	Mat gImg = gSplit(src);
	Mat rImg = rSplit(src);
	Mat dst = bImg;
	for(int i=0;i<src.rows;i++)
	for(int j=0;j<src.cols;j++)
		dst.at<uchar>(i,j) = (rImg.at<uchar>(i,j)+gImg.at<uchar>(i,j)+ bImg.at<uchar>(i,j))/3;

	return dst;
}

