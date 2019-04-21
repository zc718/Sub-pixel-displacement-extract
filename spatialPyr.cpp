#include "../include/spatialPyr.hpp"

using namespace std;
using namespace cv;

spatialPyr::spatialPyr(Size srcSize, int numLevel) :pyrLevel(numLevel) 
{
	srcHeight = srcSize.height;
	srcWidth = srcSize.width;
	meshgridX = Mat::zeros(srcHeight, srcWidth, CV_32FC1);
	meshgridY = Mat::zeros(srcHeight, srcWidth, CV_32FC1);
	for (int i = 0; i < 10; i++) 
	{
		pyrFilters.push_back(Mat(srcSize, CV_32FC2, Scalar::all(0)));
	}
}

void spatialPyr::buildLevel(Mat &srcImg, int filterIDX, Mat &dstImg) 
{
	//空间滤波器与图像频域卷积
	//multiply(srcImg, pyrFilters[filterIDX], dstImg);
	mulSpectrums(srcImg, pyrFilters[filterIDX], dstImg, 0);

	vector<Mat> spatialplanes;
	split(dstImg, spatialplanes);

	//https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
	//ifftshift
	for (size_t i = 0; i < spatialplanes.size(); i++)
	{
		// 重组傅里叶变换序列
		int cx = spatialplanes[i].cols / 2;
		int cy = spatialplanes[i].rows / 2;

		Mat q0(spatialplanes[i], Rect(0, 0, cx, cy));   // 左上
		Mat q1(spatialplanes[i], Rect(cx, 0, cx, cy));  // 右上
		Mat q2(spatialplanes[i], Rect(0, cy, cx, cy));  // 左下
		Mat q3(spatialplanes[i], Rect(cx, cy, cx, cy)); // 右下

		Mat tmp;                           // 交换象限 (左上和右下)
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);                    // 交换象限 (右上和左下)
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}
	merge(spatialplanes, dstImg);
	idft(dstImg, dstImg, DFT_COMPLEX_OUTPUT | DFT_SCALE);
}

//创建网格
void spatialPyr::meshgrid(int height, int width) 
{
	vector<float> rowX(width);
	vector<float> colY(height);
	for (int i = 0; i < width; i++) 
	{
		rowX[i] = i;
	}
	for (int j = 0; j < height; j++) 
	{
		colY[j] = j;
	}
	repeat(Mat(rowX).reshape(1, 1), height, 1, meshgridX);
	repeat(Mat(colY).reshape(1, height), 1, width, meshgridY);
}

void spatialPyr::getPolarGrid() 
{
	// 创建矩形矩阵，meshgridX、meshgridY类似于matlab的xramp,yramp
	subtract(meshgridX, (srcWidth - 0.0) / 2 * Mat::ones(srcHeight, srcWidth, CV_32FC1), meshgridX);
	meshgridX.convertTo(meshgridX, -1, 2.0 / srcWidth);
	subtract(meshgridY, (srcHeight - 0.0) / 2 * Mat::ones(srcHeight, srcWidth, CV_32FC1), meshgridY);
	meshgridY.convertTo(meshgridY, -1, 2.0 / srcWidth);

	// 笛卡尔坐标系转极坐标系
	cartToPolar(meshgridX, meshgridY, magnitudeGrid, angleGrid);
	magnitudeGrid.at<float>(srcHeight / 2, srcWidth / 2) = magnitudeGrid.at<float>(srcHeight / 2, srcWidth / 2 - 1);
}

void spatialPyr::getRadialMaskPair(double r, double twidth) 
{
	// 同matlab计算log2(rad)-log2(r)
	Mat tmpBase2 = 2.0 * Mat::ones(srcHeight, srcWidth, CV_32FC1);
	log(tmpBase2, tmpBase2);	// log2
	Mat magnitudeGrid_copy;
	magnitudeGrid.convertTo(magnitudeGrid_copy, -1, 1 / r);		// log(magnitudeGrid) - log(r) = log(magnitudeGrid/r)
	log(magnitudeGrid_copy, hiMask);
	divide(hiMask, tmpBase2, hiMask);	 // log_b(a) = log_c(a) / log_c(b)

	// Clip
	hiMask.setTo(-twidth, hiMask < -twidth);
	hiMask.setTo(0, hiMask > 0);
	hiMask.convertTo(hiMask, -1, M_PI / (2 * twidth));

	// himask = abs(cos(himask)),lomask = sqrt(1-himask.^2)
	polarToCart(Mat(), hiMask, hiMask, loMask);
	hiMask = abs(hiMask);
	loMask = abs(loMask);
}

void spatialPyr::getAngleMask(int b, int orientations) 
{
	int order = orientations - 1;

	// 计算Scaling constant
	int16_t order_factorial1 = 1;
	for (int i = 1; i < order + 1; i++)
	{
		order_factorial1 *= i;
	}
	int order_factorial2 = order_factorial1;
	for (int i = order + 1; i < 2 * order + 1; i++)
	{
		order_factorial2 *= i;
	}
	double scale_const = pow(2, (2 * order)) * pow(order_factorial1, 2) / (orientations * order_factorial2);

	// Mask angle mask
	Mat angleGrid_copy(angleGrid.size(), angleGrid.type());
	double pta;
	for (int i = 0; i < srcHeight; i++)
	{
		float *pts3 = angleGrid_copy.ptr<float>(i);
		for (int j = 0; j < srcWidth; j++)
		{
			pta = fmod(M_PI + angleGrid.ptr<float>(i)[j] - M_PI*(b - 1) / orientations, 2 * M_PI) - M_PI;
			pts3[j] = pta;
		}
	}

	// Make falloff smooth
	Mat falloff;
	threshold(abs(angleGrid_copy), falloff, M_PI_2, 1, THRESH_BINARY_INV);
	Mat tmp = Mat::ones(angleGrid.size(), angleGrid.type());
	polarToCart(Mat(), angleGrid_copy, angleGrid_copy, tmp);	// Get cos(angleGrid)
	pow(angleGrid_copy, order, angleGrid_copy);
	multiply(angleGrid_copy, falloff, angleMask, 2 * sqrt(scale_const), -1);
}

void spatialPyr::getFilters(vector<double> const &rVals, int orientations, int twidth = 1) 
{
	vector<Mat> filters((rVals.size() - 1) * orientations + 2);

	//构建空间滤波器pyrFilters
	int count = 0;
	getRadialMaskPair(rVals[0], twidth);
	filters[count] = hiMask.clone();
	count++;
	Mat lomaskPrev = loMask.clone();
	for (int i = 1; i < rVals.size(); i++) 
	{
		getRadialMaskPair(rVals[i], twidth);
		Mat radMask;
		multiply(hiMask, lomaskPrev, radMask);
		for (int j = 1; j < orientations + 1; j++) 
		{
			getAngleMask(j, orientations);
			multiply(radMask, angleMask / 2, filters[count]);
			count++;
		}
		lomaskPrev = loMask.clone();
	}
	filters[count] = loMask.clone();

	for (int i = 0; i < count - 1; i++)
	{
		// 滤波器2通道
		Mat tmp[2] = { filters[i], filters[i] };
		merge(tmp, 2, pyrFilters[i]);
	}
	
	meshgridX.release();
	meshgridY.release();
	magnitudeGrid.release();
	angleGrid.release();
	hiMask.release();
	loMask.release();
	angleMask.release();
}

void spatialPyr::octaveFilter()
{
	//四个方向的octave Filter
	meshgrid(srcHeight, srcWidth);
	getPolarGrid();
	vector<double> rVals(pyrLevel);
	for (int i = 0; i < pyrLevel; i++)
	{
		rVals[i] = pow(2, -i);
	}
	getFilters(rVals, 4);
}
