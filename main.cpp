#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <time.h>

#include "preProcessor.hpp"
#include "spatialPyr.hpp"
#include "magnifier.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap("./ruler-768��800-200fps.avi");

	//���������ӣ�������
	int scale = 2;
	//���ص�
	int ptx = 512;
	int pty = 60;

	int imgWidth;
	int imgHeight;

	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video camera" << endl;
		cin.get();
		return -1;
	}

	imgWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	imgHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	
	int length = cap.get(CV_CAP_PROP_FRAME_COUNT);

	Mat frame;
	Mat bgrImg;
	Mat bgrFloat;

	Mat vidDFT;

	Mat yiqChannels[3];
	Mat bgrChannels[3];
	// һ�����ڴ��dft�任��ʵ����һ�����ڴ���鲿, ��ʼ��ʱ��, ʵ������ͼ����, �鲿ȫΪ��

	magnifier mag(Size2i(imgWidth / pow(2, scale-1), imgHeight / pow(2, scale-1)), 2);
	preProcessor preP(bgrFloat);

	Mat RefFrame;
	Mat pyrCurrent[2];
	Mat pyrRef;
	int level = 1;
	int frameIdx = 0;

	Mat delta;
	

	//��ʱ
	clock_t start = clock();

	//���ݴ���
	vector<double> displacement;
	Mat acc;
	Mat accfft;
	Mat accelerate;
	Mat planes[2];
	Mat Altitude;

	while (cap.isOpened())
	{
		cap >> frame;
		

		//���ĳ֡Ϊ�����˳�ѭ��
		if (frame.empty())
		{
			cout << "��ʱ��"<< clock() - start << endl;
			waitKey(10);
			break;
		}
		
		for (int i = 0; i < scale-1; i++)
		{
			pyrDown(frame, frame);
		}

		frame.copyTo(bgrImg);
		//ͼ��Ԥ����(�Ҷ�ͼ)
		bgrImg.convertTo(bgrFloat, CV_32FC3);
		split(bgrFloat, yiqChannels);
		vector<Mat> complexImg = { Mat::zeros(bgrFloat.size(), CV_32FC1), Mat::zeros(bgrFloat.size(), CV_32FC1) };
		complexImg[0] = yiqChannels[0];
		preP.vid2DFT(complexImg, vidDFT);
		preP.FFTshift(vidDFT, vidDFT);
		
		//cout << vidDFT << endl;

		//��ʾƵ��ͼ
		//Mat planes[2];
		//split(vidDFT, planes);
		//magnitude(planes[0], planes[1], planes[0]);
		//Mat mag1 = planes[0];
		//mag1 += Scalar::all(1);
		//log(mag1, mag1);
		//mag1 = mag1(Rect(0, 0, mag1.cols & -2, mag1.rows & -2));
		//normalize(mag1, mag1, 0, 1, CV_MINMAX);
		//imshow("before rearrange ", mag1);
		//if (waitKey(10) == 'q')
		//break;

		if (frameIdx == 0)
		{
			
			//����ο�֡����λ
			mag.spatialFilter.buildLevel(vidDFT, level, RefFrame); 
			//��ʾƵ��ͼ
			//Mat planes[2];
			//split(RefFrame, planes);
			//phase(planes[0], planes[1], planes[0]);
			//Mat mag1 = planes[0];
			//mag1 += Scalar::all(1);
			//log(mag1, mag1);
			//mag1 = mag1(Rect(0, 0, mag1.cols & -2, mag1.rows & -2));
			//normalize(mag1, mag1, 0, 1, CV_MINMAX);
			//imshow("before rearrange ", mag1);
			//if (waitKey(10) == 'q')
			//break;
			split(RefFrame, pyrCurrent); 
			phase(pyrCurrent[0], pyrCurrent[1], pyrCurrent[1]);
			pyrRef = pyrCurrent[1];		
		}
		else
		{
			//������λ��
			mag.maginify(vidDFT, level, pyrRef, scale, delta);
			//cout << delta.at<float>(pty / scale, ptx / scale) << endl;
			//a[frameIdx - 1] = delta.at<float>(pty / scale, ptx / scale);
			//cout << a[frameIdx - 1] << endl;
			displacement.push_back(delta.at<float>(pty / scale, ptx / scale));
			cout << displacement[frameIdx-1] << endl;
		} 
	    frameIdx++;
	}
	//displacement = Mat(displacement, true);

	//GaussianBlur(displacement, displacement, Size(3, 3), 0);
	//Laplacian(displacement, acc, CV_64FC1, 1);
	//vector<Mat> acccomplex = { Mat::zeros(acc.size(), CV_64FC1), Mat::zeros(acc.size(), CV_64FC1) };

	//acccomplex[0] = acc;
	//merge(acccomplex, accelerate);
	//dft(accelerate, accfft, DFT_COMPLEX_OUTPUT);

	//split(accfft, planes);
	//magnitude(planes[0], planes[1], planes[0]);
	//Altitude = planes[0];

	//for (int i = 0; i < 600; i++)
	//{
	//	cout << acc.at<double>(i) << endl;
	//}

	return 0;
}

