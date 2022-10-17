// CMLPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "CMLPR.h"

using namespace cv;

Mat RGBToGray(Mat rgb)
{
	int stride = 3;
	Mat gray = Mat::zeros(rgb.size(), CV_8UC1);

	for (size_t i = 0; i < rgb.rows; i++)
	{
		for (size_t j = 0; j < rgb.cols * stride; j += stride)
		{
			auto r = rgb.at<uchar>(i, j);
			auto g = rgb.at<uchar>(i, j + 1);
			auto b = rgb.at<uchar>(i, j + 2);

			gray.at<uchar>(i, j / stride) = (r + g + b) / stride;
		}
	}

	return gray;
}

Mat RGBToBinary(Mat rgb, int threshold = 128)
{
	int stride = 3;
	Mat binary = Mat::zeros(rgb.size(), CV_8UC1);

	for (size_t i = 0; i < rgb.rows; i++)
	{
		for (size_t j = 0; j < rgb.cols * stride; j += stride)
		{
			auto r = rgb.at<uchar>(i, j);
			auto g = rgb.at<uchar>(i, j + 1);
			auto b = rgb.at<uchar>(i, j + 2);
			auto result = (r + g + b) / stride;

			if (result > threshold)
				binary.at<uchar>(i, j / stride) = 255;
			//binary.at<uchar>(i, j / stride) = result > 128 ? 255 : 0;
		}
	}

	return binary;
}

Mat GrayToBinary(Mat gray, int threshold = 128)
{
	Mat binary = Mat::zeros(gray.size(), CV_8UC1);

	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j += 1)
		{
			if (gray.at<uchar>(i, j) > threshold)
				binary.at<uchar>(i, j) = 255;
			//binary.at<uchar>(i, j) = gray.at<uchar>(i, j) > 128 ? 255 : 0;
		}
	}

	return binary;
}

Mat GrayInversion(Mat gray)
{
	Mat inversion = Mat::zeros(gray.size(), CV_8UC1);

	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j += 1)
		{
			inversion.at<uchar>(i, j) = 255 - gray.at<uchar>(i, j);
		}
	}

	return inversion;
}

Mat GrayStep(Mat gray, int minThreshold = 80, int maxThreshold = 140)
{
	Mat step = Mat::zeros(gray.size(), CV_8UC1);

	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j++)
		{
			auto temp = gray.at<uchar>(i, j);

			if (temp >= minThreshold && temp <= maxThreshold)
				step.at<uchar>(i, j) = 255;
		}
	}
	return step;
}

Mat GrayAverage3x3(Mat gray)
{
	Mat average = Mat::zeros(gray.size(), CV_8UC1);

	for (int i = 1; i < gray.rows - 1; i++)
	{
		for (int j = 1; j < gray.cols - 1; j++)
		{
			for (int q = -1; q < 2; q++)
			{
				for (int z = -1; z < 2; z++)
				{
					average.at<uchar>(i, j) += (gray.at<uchar>(i + q, j + z)) / 9;
				}
			}
		}
	}
	return average;
}

int GetSum(Mat& gray, int n, int i, int j)
{
	int sum = 0;
	auto th = ((n - 1) / 2);
	for (int q = -th; q <= th; q++)
	{
		for (int z = -th; z <= th; z++)
		{
			sum += gray.at<uchar>(i + q, j + z);
		}
	}
	return sum;
}

Mat AverageNxN(Mat gray, int n)
{
	Mat average = Mat::zeros(gray.size(), CV_8UC1);

	for (int i = 1; i < gray.rows - 1; i++)
	{
		for (int j = 1; j < gray.cols - 1; j++)
		{
			auto sum = GetSum(gray, n, i, j);
			average.at<uchar>(i, j) += sum / (n * n);
		}
	}
	return average;
}

Mat Avg(Mat Grey, int neighbirSize)
{
	Mat AvgImg = Mat::zeros(Grey.size(), CV_8UC1);
	int totalPix = pow(2 * neighbirSize + 1, 2);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int sum = 0;
			int count = 0;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					count++;
					sum += Grey.at<uchar>(i + ii, j + jj);
				}
			}
			AvgImg.at<uchar>(i, j) = sum / count;
		}
	}

	return AvgImg;
}

Mat Max(Mat Grey, int neighbirSize)
{
	Mat img = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int max = -1;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					int pixel = Grey.at<uchar>(i + ii, j + jj);
					if (pixel > max)
						max = pixel;
				}
			}
			img.at<uchar>(i, j) = max;
		}
	}
	return img;
}

Mat Min(Mat Grey, int neighbirSize)
{
	Mat img = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int min = 255;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					int pixel = Grey.at<uchar>(i + ii, j + jj);
					if (pixel < min)
						min = pixel;
				}
			}
			img.at<uchar>(i, j) = min;
		}
	}
	return img;
}

Mat Edge(Mat Grey, int th)
{
	Mat EdgeImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int AvgL = (Grey.at<uchar>(i - 1, j - 1) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i + 1, j - 1)) / 3;
			int AvgR = (Grey.at<uchar>(i - 1, j + 1) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1)) / 3;
			if (abs(AvgL - AvgR) > th)
				EdgeImg.at<uchar>(i, j) = 255;


		}
	}

	return EdgeImg;


}


Mat Dialation(Mat edge, int neighbirSize)
{

	Mat dialation = Mat::zeros(edge.size(), CV_8UC1);

	for (int i = neighbirSize; i < edge.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < edge.cols - neighbirSize; j++)
		{
			auto isPixelBlack = edge.at<uchar>(i, j) == 0;
			if (!isPixelBlack)
			{
				dialation.at<uchar>(i, j) = 255;
				continue;
			}

			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					auto isNeighbourWhite = edge.at<uchar>(i + ii, j + jj) == 255;
					if (isNeighbourWhite)
					{
						dialation.at<uchar>(i, j) = 255;
						break;
					}

				}
			}
		}
	}

	return dialation;
}

Mat Erosion(Mat edge, int neighbirSize)
{

	Mat erosion = Mat::zeros(edge.size(), CV_8UC1);

	for (int i = neighbirSize; i < edge.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < edge.cols - neighbirSize; j++)
		{

			erosion.at<uchar>(i, j) = 255;
			auto isPixelWhite = edge.at<uchar>(i, j) == 255;
			if (!isPixelWhite)
			{
				erosion.at<uchar>(i, j) = 0;
				continue;
			}

			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					auto isNeighbourBlack = edge.at<uchar>(i + ii, j + jj) == 0;
					if (isNeighbourBlack)
					{
						erosion.at<uchar>(i, j) = 0;
						break;
					}

				}
			}
		}
	}

	return erosion;
}

int main()
{
	Mat img;
	img = imread("C:\\Users\\L B O\\Downloads\\1.jpg");
	imshow("RGB Image", img);

	auto gray = RGBToGray(img);
	imshow("gray Image", gray);

	auto binary = GrayToBinary(gray);
	imshow("binary Image", binary);

	auto inversion = GrayInversion(gray);
	imshow("gray Image Inverted", inversion);

	auto step = GrayStep(gray);
	imshow("gray Image Step", step);

	auto average = AverageNxN(gray, 3);
	imshow("gray Image average", average);

	auto max = Max(gray, 3);
	imshow("gray Image Max", max);

	auto edge = Edge(average, 50);
	imshow("Edge", edge);

	auto erosion = Erosion(edge, 1);
	imshow("Erosion", erosion);

	auto dialation = Dialation(erosion, 15);
	imshow("Dialation", dialation);


	waitKey();
	std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
