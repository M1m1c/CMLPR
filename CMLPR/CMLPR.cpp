// CMLPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <baseapi.h>
#include <allheaders.h>

#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "CMLPR.h"

using namespace cv;
using namespace std;

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
			bool shouldBreak = false;

			for (int ii = -4; ii <= 4; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					auto isNeighbourWhite = edge.at<uchar>(i + ii, j + jj) == 255;
					if (isNeighbourWhite)
					{
						dialation.at<uchar>(i, j) = 255;
						shouldBreak = true;
						break;
					}

				}

				if (shouldBreak) { break; }
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
			int blackNeighbors = 0;
			erosion.at<uchar>(i, j) = edge.at<uchar>(i, j);
			bool shouldBreak = false;

			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					auto isNeighbourBlack = edge.at<uchar>(i + ii, j + jj) == 0;
					if (isNeighbourBlack)
					{
						blackNeighbors++;
						// erosion.at<uchar>(i, j) = 0;
						// shouldBreak = true;
						// break;
					}

				}
				// if (shouldBreak) { break; }
			}
			if (blackNeighbors > 3)
			{
				erosion.at<uchar>(i, j) = 0;
			}
		}
	}

	return erosion;
}

Mat EraseBorder(Mat edge, int borderX, int borderY)
{

	Mat retImg = Mat::zeros(edge.size(), CV_8UC1);

	for (int i = borderY; i < edge.rows - borderY; i++)
	{
		for (int j = borderX; j < edge.cols - borderX; j++)
		{

			retImg.at<uchar>(i, j) = edge.at<uchar>(i, j);

		}
	}

	return retImg;
}


int ContrastValue(int value)
{
	float f = value;
	f /= 255.f;

	f = pow(f, 3.5f);
	f *= 255.f;
	return (int)f;
}

Mat ContrastImg(Mat img)
{
	Mat retval = Mat::zeros(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int val = ContrastValue(img.at<uchar>(i, j));
			img.at<uchar>(i, j) = val;
		}
	}
	return img;
}

Mat EqHist(Mat gray)
{
	Mat eqImg = Mat::zeros(gray.size(), CV_8UC1);

	int count[256] = { 0 };
	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j++)
		{
			count[gray.at<uchar>(i, j)]++;
		}
	}

	float prob[256] = { 0.0 };
	for (size_t i = 0; i < 256; i++)
	{
		prob[i] = (float)count[i] / (float)(gray.rows * gray.cols);
	}

	float accprob[256] = { 0.0 };
	accprob[0] = prob[0];
	for (size_t i = 1; i < 256; i++)
	{
		accprob[i] = prob[i] + accprob[i - 1];
	}

	float newValue[256] = { 0.0 };
	for (size_t i = 0; i < 256; i++)
	{
		newValue[i] = 255 * accprob[i];
	}

	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j++)
		{
			eqImg.at<uchar>(i, j) = newValue[gray.at<uchar>(i, j)];
		}
	}
	return eqImg;
}

int OTSU(Mat Grey)
{
	int count[256] = { 0 };
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			count[Grey.at<uchar>(i, j)]++;


	// prob
	float prob[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);

	// accprob
	float theta[256] = { 0.0 };
	theta[0] = prob[0];
	for (int i = 1; i < 256; i++)
		theta[i] = prob[i] + theta[i - 1];

	float meu[256] = { 0.0 };
	for (int i = 1; i < 256; i++)
		meu[i] = i * prob[i] + meu[i - 1];

	float sigma[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		sigma[i] = pow(meu[255] * theta[i] - meu[i], 2) / (theta[i] * (1 - theta[i]));

	int index = 0;
	float maxVal = 0;
	for (int i = 0; i < 256; i++)
	{
		if (sigma[i] > maxVal)
		{
			maxVal = sigma[i];
			index = i;
		}
	}

	return index + 30;
}
void showAll()
{
	Mat images[20];
	images[0] = imread("..\\Dataset\\1.jpg");
	images[1] = imread("..\\Dataset\\2.jpg");
	images[2] = imread("..\\Dataset\\3.jpg");
	images[3] = imread("..\\Dataset\\4.jpg");
	images[4] = imread("..\\Dataset\\5.jpg");
	images[5] = imread("..\\Dataset\\6.jpg");
	images[6] = imread("..\\Dataset\\7.jpg");
	images[7] = imread("..\\Dataset\\8.jpg");
	images[8] = imread("..\\Dataset\\9.jpg");
	images[9] = imread("..\\Dataset\\10.jpg");
	images[10] = imread("..\\Dataset\\11.jpg");
	images[11] = imread("..\\Dataset\\12.jpg");
	images[12] = imread("..\\Dataset\\13.jpg");
	images[13] = imread("..\\Dataset\\14.jpg");
	images[14] = imread("..\\Dataset\\15.jpg");
	images[15] = imread("..\\Dataset\\16.jpg");
	images[16] = imread("..\\Dataset\\17.jpg");
	images[17] = imread("..\\Dataset\\18.jpg");
	images[18] = imread("..\\Dataset\\19.jpg");
	images[19] = imread("..\\Dataset\\20.jpg");

	Mat img;
	img = imread("..\\Dataset\\8.jpg");
	// imshow("RGB Image", img);

	auto gray = RGBToGray(img);
	// imshow("gray Image", gray);

	for (int i = 0; i < 20; i++)
	{
		Mat image = images[i];
		image = RGBToGray(image);
		Mat avg = AverageNxN(image, 1);
		Mat edge = Edge(avg, 50);
		Mat eroded = Erosion(edge, 1);
		Mat dilated = Dialation(eroded, 5);

		Mat DilatedImgCpy;
		DilatedImgCpy = dilated.clone();
		vector<vector<Point>> contours1;
		vector<Vec4i> hierachy1;
		findContours(dilated, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
		Mat dst = Mat::zeros(gray.size(), CV_8UC3);

		Rect rect;
		Mat plate;
		Scalar black = CV_RGB(0, 0, 0);
		for (int ii = 0; ii < contours1.size(); ii++)
		{
			rect = boundingRect(contours1[i]);

			auto ratio = (float)rect.width / (float)rect.height;

			auto tooTall = rect.height > 100;
			auto tooWide = rect.width < 70 || rect.width > 400;
			auto  outsideFocusX = rect.x < 0.15 * DilatedImgCpy.cols || rect.x > 0.85 * DilatedImgCpy.cols;
			auto  outsideFocusY = rect.y < 0.3 * DilatedImgCpy.rows || rect.y > 0.85 * DilatedImgCpy.rows;
			if (tooTall || tooWide || outsideFocusX || outsideFocusY || ratio < 1.5f)
			{
				drawContours(DilatedImgCpy, contours1, i, black, -1, 8, hierachy1);
			}
			else
			{
				plate = gray(rect);
			}

		}
		string title = std::to_string(i);
		imshow(title, dilated);
		// imshow("Filtered Image ", DilatedImgCpy);

		title += title;
		if (plate.cols != 0 && plate.rows != 0)
			imshow(title, plate);
	}

}

Mat LocateLicensePlate(Mat image)
{

	Mat gray = RGBToGray(image);
	//imshow("Grey image", gray);

	auto average = AverageNxN(gray, 1);
	// imshow("gray Image average", average);

	auto edge = Edge(average, 50);
	//imshow("Edge", edge);

	auto erosion = Erosion(edge, 1);
	//imshow("Erosion", erosion);

	auto dialation = Dialation(erosion, 5);
	//imshow("Dialation", dialation);



	Mat DilatedImgCpy;
	DilatedImgCpy = dialation.clone();
	vector<vector<Point>> contours1;
	vector<Vec4i> hierachy1;
	findContours(dialation, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	Mat dst = Mat::zeros(gray.size(), CV_8UC3);


	Rect rect;
	Mat plate;
	Scalar black = CV_RGB(0, 0, 0);
	for (int i = 0; i < contours1.size(); i++)
	{
		rect = boundingRect(contours1[i]);

		auto ratio = (float)rect.width / (float)rect.height;

		auto tooTall = rect.height > 100;
		auto tooWide = rect.width < 70 || rect.width > 400;
		auto  outsideFocusX = rect.x < 0.15 * DilatedImgCpy.cols || rect.x > 0.85 * DilatedImgCpy.cols;
		auto  outsideFocusY = rect.y < 0.3 * DilatedImgCpy.rows || rect.y > 0.85 * DilatedImgCpy.rows;
		if (tooTall || tooWide || outsideFocusX || outsideFocusY || ratio < 1.5f)
		{
			drawContours(DilatedImgCpy, contours1, i, black, -1, 8, hierachy1);
		}
		else
		{
			plate = gray(rect);
		}

	}

	return plate;
}

string ProcessLicensePlate(Mat plate, int i)
{
	string str = "NO LETTERS DETECTED!";

	//TODO linear remapping

	plate = ContrastImg(plate);
	imshow("funcPlate", plate);

	//plate = EqHist(plate);

	int OTSUTH = OTSU(plate);

	Mat BinPlate = GrayToBinary(plate, OTSUTH);

	BinPlate = EraseBorder(BinPlate, ((float)BinPlate.cols * 0.05f), ((float)BinPlate.rows * 0.20f));
	

	//BinPlate = Dialation(BinPlate, 1);
	//imshow("BinPlate", BinPlate);
	imshow(to_string(i + 1), BinPlate);
	tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

	api->Init("..\\tessdata", "eng");

	cvtColor(BinPlate, BinPlate, COLOR_BGR2BGRA);
	api->SetImage(BinPlate.data, BinPlate.cols, BinPlate.rows, 4, 4 * BinPlate.cols);

	char* outText = api->GetUTF8Text();
	if (outText != nullptr)
	{
		str = outText;
		str.erase(remove(str.begin(), str.end(), '('), str.end());
		str.erase(remove(str.begin(), str.end(), ')'), str.end());
		str.erase(remove(str.begin(), str.end(), '|'), str.end());
		str.erase(remove(str.begin(), str.end(), ' '), str.end());
		str.erase(remove(str.begin(), str.end(), '.'), str.end());
		str.erase(remove(str.begin(), str.end(), '['), str.end());
		str.erase(remove(str.begin(), str.end(), ']'), str.end());
		str.erase(remove(str.begin(), str.end(), '{'), str.end());
		str.erase(remove(str.begin(), str.end(), '}'), str.end());
		str.erase(remove(str.begin(), str.end(), '"'), str.end());
		str.erase(remove(str.begin(), str.end(), '_'), str.end());
		str.erase(remove(str.begin(), str.end(), '	'), str.end());
	}
	api->End();
	delete[] outText;

	return toUpperCase(str);
}

// 8, 10, 14, 16, 18, 20
int main()
{

	Mat images[20];
	images[0] = imread("..\\Dataset\\1.jpg");
	images[1] = imread("..\\Dataset\\2.jpg");
	images[2] = imread("..\\Dataset\\3.jpg");
	images[3] = imread("..\\Dataset\\4.jpg");
	images[4] = imread("..\\Dataset\\5.jpg");
	images[5] = imread("..\\Dataset\\6.jpg");
	images[6] = imread("..\\Dataset\\7.jpg");
	images[7] = imread("..\\Dataset\\8.jpg");
	images[8] = imread("..\\Dataset\\9.jpg");
	images[9] = imread("..\\Dataset\\10.jpg");
	images[10] = imread("..\\Dataset\\11.jpg");
	images[11] = imread("..\\Dataset\\12.jpg");
	images[12] = imread("..\\Dataset\\13.jpg");
	images[13] = imread("..\\Dataset\\14.jpg");
	images[14] = imread("..\\Dataset\\15.jpg");
	images[15] = imread("..\\Dataset\\16.jpg");
	images[16] = imread("..\\Dataset\\17.jpg");
	images[17] = imread("..\\Dataset\\18.jpg");
	images[18] = imread("..\\Dataset\\19.jpg");
	images[19] = imread("..\\Dataset\\20.jpg");


	for (size_t i = 0; i < 20; i++)
	{
		Mat plate = LocateLicensePlate(images[i]);
		string str = "PLATE NOT FOUND!";
		if (plate.cols != 0 && plate.rows != 0)
		{
			str = ProcessLicensePlate(plate, i);
			//imshow(to_string(i+1), plate);
		}
		cout << i+1 << ": " << str << endl;//ProcessLicensePlate(plate);
	}

	//Mat image = imread("..\\Dataset\\2.jpg");

	//Mat plate = LocateLicensePlate(image);
	//if (plate.cols != 0 && plate.rows != 0)
	//{
	//	cout << "OCR output:\n" << ProcessLicensePlate(plate);  //printf("OCR output:\n%s", str);
	//}

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
