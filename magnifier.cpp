#include "../include/magnifier.hpp"

using namespace std;
using namespace cv;


magnifier::magnifier(Size2i imgSize, int numLevel) :
spatialFilter(imgSize, numLevel)
{
	spatialFilter.octaveFilter();
	intModMat = Mat::zeros(imgSize, CV_8UC2);
	floatModMat = Mat::zeros(imgSize, CV_32FC2);
}

void magnifier::magnifyLevel(Mat &srcImg, int levelIDX, Mat pyrRef, int scale, Mat &delta)
{
	spatialFilter.buildLevel(srcImg, levelIDX, curLevelFrame); 
	split(curLevelFrame, pyrCurrent); 
	
	// 求相位差
	phase(pyrCurrent[0], pyrCurrent[1], pyrCurrent[1]);
	deltaCurrent = pyrCurrent[1] - pyrRef;
	matMod(deltaCurrent, deltaCurrent);

	//求幅值
	magnitude(pyrCurrent[0], pyrCurrent[1], pyrCurrent[1]);
	phaseMagnitude = pyrCurrent[1];
	multiply(deltaCurrent, phaseMagnitude, deltaCurrent);
	GaussianBlur(deltaCurrent, deltaCurrent, Size(11, 11), 3);
	GaussianBlur(phaseMagnitude, phaseMagnitude, Size(11, 11), 3);
	divide(deltaCurrent, phaseMagnitude, delta);
	//cout << deltaCurrent.at<float>(pty / scale, ptx / scale) << endl;
}

// srcImg: fft2 of input video frame
void magnifier::maginify(Mat &srcImg, int levelIDX, Mat pyrRef, int scale, Mat &delta)
{
	magnifyLevel(srcImg, levelIDX, pyrRef, scale, delta);
}

void magnifier::matMod(Mat &srcImg, Mat &dstImg)
{
	srcImg.convertTo(intModMat, CV_8UC2, M_1_PI / 2);  // int(a/(2*PI))
	intModMat.convertTo(floatModMat, CV_32FC2, 2 * M_PI);
	subtract(srcImg, floatModMat, dstImg);
}