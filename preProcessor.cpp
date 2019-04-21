#include "../include/preProcessor.hpp"

using namespace std;
using namespace cv;

preProcessor::preProcessor(Mat &srcImg)
{

}

void preProcessor::vid2DFT(vector<Mat> &srcImg, Mat &dstImg)
{
	// 图像fft
	merge(srcImg, dstImg);
	dft(dstImg, dstImg, DFT_COMPLEX_OUTPUT);
}

void preProcessor::FFTshift(Mat &dstImg, Mat &out)
{
	vector<Mat> planes;
	split(dstImg, planes);

	for (size_t i = 0; i < planes.size(); i++)
	{
		// 重组傅里叶变换序列
		int cx = planes[i].cols / 2;
		int cy = planes[i].rows / 2;

		Mat q0(planes[i], Rect(0, 0, cx, cy));   
		Mat q1(planes[i], Rect(cx, 0, cx, cy));  
		Mat q2(planes[i], Rect(0, cy, cx, cy));  
		Mat q3(planes[i], Rect(cx, cy, cx, cy)); 

		Mat tmp;             
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}
	merge(planes, out);
}
