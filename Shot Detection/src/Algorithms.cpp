#include "Algorithms.hpp"

#include <vector>
#include <map>
#include <filesystem>

template <typename T>
T absDifference(T a, T b) {
	return a >= b ? a - b : b - a;
}


void genCutFrames(const std::string& filepath, const std::vector<int>& cuts, const std::vector<cv::Mat>& frames)
{
	for (int i = 0; i < cuts.size(); i++) {
		std::string imgDir = filepath + "/" + std::to_string(cuts[i]);
		std::filesystem::create_directories(imgDir);
		for (int j = std::max(0, cuts[i] - 5); j < std::min((int)frames.size(), cuts[i] + 5); j++)
			cv::imwrite(imgDir + "/" + std::to_string(j) + ".png", frames[j]);
	}
}

void PixelDifference::algo()
{
	std::filesystem::remove_all(m_FrameFilePath);

	cv::Mat frame, grayFrame;
	std::vector<cv::Mat> allFrames;
	std::vector<int> cuts;
	uint8_t* prevIntensity;

	// First Frame
	getVideo().read(frame);
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
	prevIntensity = new uint8_t[grayFrame.rows * grayFrame.cols];
	uint8_t* imageData = (uint8_t*)grayFrame.data;
	for (int i = 0; i < grayFrame.rows; i++) {
		for (int j = 0; j < grayFrame.cols; j++) {
			prevIntensity[i * grayFrame.cols + j] = imageData[i * grayFrame.cols + j];
		}
	}
	allFrames.push_back(frame.clone());
	
	uint frame_index = 2;
	while (true) {
		if (!getVideo().read(frame)) break;

		cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

		double dIntensity = 0;

		uint8_t* imageData = (uint8_t*)grayFrame.data;
		for (int i = 0; i < grayFrame.rows; i++) {
			for (int j = 0; j < grayFrame.cols; j++) {
				uint8_t prevI = prevIntensity[i * grayFrame.cols + j];
				uint8_t currI = imageData[i * grayFrame.cols + j];

				dIntensity += absDifference(prevI, currI);

				prevIntensity[i * grayFrame.cols + j] = currI;
			}
		}


		if (frame_index != 1) {
			m_ResultFile << frame_index << "\t\t\t" << dIntensity << '\n';

			if (dIntensity > m_Threshold) {
				cuts.push_back(frame_index-1);
			}
		}
		
		allFrames.push_back(frame.clone());
		frame_index++;
	}

	m_TotalFrames = frame_index;

	genCutFrames(m_FrameFilePath, cuts, allFrames);

	delete prevIntensity;
}

void PixelDifferenceColor::algo()
{
	std::filesystem::remove_all(m_FrameFilePath);

	cv::Mat frame;
	std::vector<cv::Mat> allFrames;
	std::vector<int> cuts;

	struct Intensity {
		uint8_t *b, *g, *r;
	} prevIntensity;

	// First Frame
	getVideo().read(frame);
	prevIntensity.b = new uint8_t[frame.rows * frame.cols];
	prevIntensity.g = new uint8_t[frame.rows * frame.cols];
	prevIntensity.r = new uint8_t[frame.rows * frame.cols];

	int channels = frame.channels();
	uint8_t* imageData = (uint8_t*)frame.data;
	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			prevIntensity.b[i * frame.cols + j] = imageData[i * frame.cols * channels + j * channels + 0];
			prevIntensity.g[i * frame.cols + j] = imageData[i * frame.cols * channels + j * channels + 1];
			prevIntensity.r[i * frame.cols + j] = imageData[i * frame.cols * channels + j * channels + 2];
		}
	}

	allFrames.push_back(frame.clone());

	uint frame_index = 2;
	while (true) {
		if (!getVideo().read(frame)) break;

		double dIntensity = 0;

		int channels = frame.channels();
		uint8_t* imageData = (uint8_t*)frame.data;
		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				uint8_t prevI_b = prevIntensity.b[i * frame.cols + j];
				uint8_t prevI_g = prevIntensity.g[i * frame.cols + j];
				uint8_t prevI_r = prevIntensity.r[i * frame.cols + j];

				uint8_t currI_b = imageData[i * frame.cols * channels + j * channels + 0];
				uint8_t currI_g = imageData[i * frame.cols * channels + j * channels + 1];
				uint8_t currI_r = imageData[i * frame.cols * channels + j * channels + 2];

				dIntensity += absDifference(prevI_b, currI_b);
				dIntensity += absDifference(prevI_g, currI_g);
				dIntensity += absDifference(prevI_r, currI_r);

				prevIntensity.b[i * frame.cols + j] = currI_b;
				prevIntensity.g[i * frame.cols + j] = currI_g;
				prevIntensity.r[i * frame.cols + j] = currI_r;
			}
		}

		if (frame_index != 1) {
			m_ResultFile << frame_index << "\t\t\t" << dIntensity << '\n';

			if (dIntensity > m_Threshold) {
				cuts.push_back(frame_index-1);
			}
		}

		allFrames.push_back(frame.clone());
		frame_index++;
	}

	m_TotalFrames = frame_index;

	genCutFrames(m_FrameFilePath, cuts, allFrames);

	delete prevIntensity.b, prevIntensity.g, prevIntensity.r;
}

void Histogram_Bin2Bin::algo()
{
	std::filesystem::remove_all(m_FrameFilePath);

	cv::Mat frame, grayFrame;
	std::vector<cv::Mat> allFrames;
	std::vector<int> cuts;

	struct ColorBins {
		uint32_t rBin[256] = { 0 }, gBin[256] = { 0 }, bBin[256] = { 0 };
	} prevBins;


	uint frame_index = 1;
	while (true) {
		if (!getVideo().read(frame)) break;

		ColorBins newBins;

		int channels = frame.channels();
		uint8_t* imageData = (uint8_t*)frame.data;
		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				uint8_t b = imageData[i*frame.cols*channels + j * channels + 0];
				uint8_t g = imageData[i*frame.cols*channels + j * channels + 1];
				uint8_t r = imageData[i*frame.cols*channels + j * channels + 2];

				newBins.bBin[b]++; newBins.gBin[g]++; newBins.rBin[r]++;
			}
		}

		double dIntensity = 0;

		for (int i = 0; i < 256; i++) {
			dIntensity += absDifference(newBins.bBin[i], prevBins.bBin[i]);
			dIntensity += absDifference(newBins.gBin[i], prevBins.gBin[i]);
			dIntensity += absDifference(newBins.rBin[i], prevBins.rBin[i]);

			prevBins.bBin[i] = newBins.bBin[i];
			prevBins.gBin[i] = newBins.gBin[i];
			prevBins.rBin[i] = newBins.rBin[i];
		}

		if (frame_index != 1) {
			double value = dIntensity / (frame.rows * frame.cols);
			m_ResultFile << frame_index << "\t\t\t" << value << '\n';
			
			if (value > m_Threshold) {
				cuts.push_back(frame_index-1);
			}
		}

		allFrames.push_back(frame.clone());
		frame_index++;
	}

	m_TotalFrames = frame_index;

	genCutFrames(m_FrameFilePath, cuts, allFrames);
}

void Histogram_ChiSqrNew::algo()
{
	std::filesystem::remove_all(m_FrameFilePath);

	cv::Mat frame, grayFrame;
	std::vector<cv::Mat> allFrames;
	std::vector<int> cuts;

	struct ColorBins {
		uint32_t rBin[256] = { 0 }, gBin[256] = { 0 }, bBin[256] = { 0 };
	} prevBins;


	uint frame_index = 1;
	while (true) {
		if (!getVideo().read(frame)) break;

		ColorBins newBins;

		int channels = frame.channels();
		uint8_t* imageData = (uint8_t*)frame.data;
		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				uint8_t b = imageData[i*frame.cols*channels + j * channels + 0];
				uint8_t g = imageData[i*frame.cols*channels + j * channels + 1];
				uint8_t r = imageData[i*frame.cols*channels + j * channels + 2];

				newBins.bBin[b]++; newBins.gBin[g]++; newBins.rBin[r]++;
			}
		}

		double dIntensity = 0;

		for (int i = 0; i < 256; i++) {
			uint32_t diff;
			if (prevBins.bBin[i] != 0) {
				diff = absDifference(newBins.bBin[i], prevBins.bBin[i]);
				dIntensity += (diff * diff) / (prevBins.bBin[i] * prevBins.bBin[i]);
			}
			if (prevBins.gBin[i] != 0) {
				diff = absDifference(newBins.gBin[i], prevBins.gBin[i]);
				dIntensity += (diff * diff) / (prevBins.gBin[i] * prevBins.gBin[i]);
			}
			if (prevBins.rBin[i] != 0) {
				diff = absDifference(newBins.rBin[i], prevBins.rBin[i]);
				dIntensity += (diff * diff) / (prevBins.rBin[i] * prevBins.rBin[i]);
			}

			prevBins.bBin[i] = newBins.bBin[i];
			prevBins.gBin[i] = newBins.gBin[i];
			prevBins.rBin[i] = newBins.rBin[i];
		}

		if (frame_index != 1) {
			double value = dIntensity;// / (frame.rows * frame.cols);
			m_ResultFile << frame_index << "\t\t\t" << value << '\n';// / (frame.rows * frame.cols * frame.rows * frame.cols)

			if (value > m_Threshold) {
				cuts.push_back(frame_index-1);
			}
			
		}
		
		allFrames.push_back(frame.clone());
		frame_index++;
	}

	m_TotalFrames = frame_index;

	genCutFrames(m_FrameFilePath, cuts, allFrames);
}


void Histogram_ChiSqrOld::algo()
{
	std::filesystem::remove_all(m_FrameFilePath);

	cv::Mat frame, grayFrame;
	std::vector<cv::Mat> allFrames;
	std::vector<int> cuts;

	struct ColorBins {
		uint32_t rBin[256] = { 0 }, gBin[256] = { 0 }, bBin[256] = { 0 };
	} prevBins;


	uint frame_index = 1;
	while (true) {
		if (!getVideo().read(frame)) break;

		ColorBins newBins;

		int channels = frame.channels();
		uint8_t* imageData = (uint8_t*)frame.data;
		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				uint8_t b = imageData[i*frame.cols*channels + j * channels + 0];
				uint8_t g = imageData[i*frame.cols*channels + j * channels + 1];
				uint8_t r = imageData[i*frame.cols*channels + j * channels + 2];

				newBins.bBin[b]++; newBins.gBin[g]++; newBins.rBin[r]++;
			}
		}

		double dIntensity = 0;

		for (int i = 0; i < 256; i++) {
			uint32_t diff;
			if (newBins.bBin[i] != 0 && prevBins.bBin[i] != 0) {
				diff = absDifference(newBins.bBin[i], prevBins.bBin[i]);
				dIntensity += (diff * diff) / std::max(newBins.bBin[i], prevBins.bBin[i]);
			}
			if (newBins.gBin[i] != 0 && prevBins.gBin[i] != 0) {
				diff = absDifference(newBins.gBin[i], prevBins.gBin[i]);
				dIntensity += (diff * diff) / std::max(newBins.gBin[i], prevBins.gBin[i]);
			}
			if (newBins.rBin[i] != 0 && prevBins.rBin[i] != 0) {
				diff = absDifference(newBins.rBin[i], prevBins.rBin[i]);
				dIntensity += (diff * diff) / std::max(newBins.rBin[i], prevBins.rBin[i]);
			}

			prevBins.bBin[i] = newBins.bBin[i];
			prevBins.gBin[i] = newBins.gBin[i];
			prevBins.rBin[i] = newBins.rBin[i];
		}

		if (frame_index != 1) {
			double value = dIntensity; // / (frame.rows * frame.cols * frame.rows * frame.cols)
			m_ResultFile << frame_index << "\t\t\t" << value << '\n'; 

			if (value > m_Threshold) {
				cuts.push_back(frame_index-1);
			}
		}

		allFrames.push_back(frame.clone());
		frame_index++;
	}

	m_TotalFrames = frame_index;

	genCutFrames(m_FrameFilePath, cuts, allFrames);
}

void Histogram_Intersect::algo()
{
	std::filesystem::remove_all(m_FrameFilePath);

	cv::Mat frame, grayFrame;
	std::vector<cv::Mat> allFrames;
	std::vector<int> cuts;

	
	uint32_t prevHist[256] = { 0 };


	uint frame_index = 1;
	while (true) {
		if (!getVideo().read(frame)) break;

		uint32_t currHist[256] = { 0 };

		int channels = frame.channels();
		uint8_t* imageData = (uint8_t*)frame.data;
		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				uint8_t b = imageData[i*frame.cols*channels + j * channels + 0];
				uint8_t g = imageData[i*frame.cols*channels + j * channels + 1];
				uint8_t r = imageData[i*frame.cols*channels + j * channels + 2];

				currHist[(b + g + r) / 3]++;
			}
		}

		double value = 0;

		for (int i = 0; i < 256; i++) {
			value += std::max(prevHist[i], currHist[i]);
			prevHist[i] = currHist[i];
		}
		if (frame_index != 1) {
			value = 1 - value / (frame.rows * frame.cols);
			m_ResultFile << frame_index << "\t\t\t" << value << '\n';
			
			if (value < m_Threshold) {
				cuts.push_back(frame_index-1);
			}
		}

		allFrames.push_back(frame.clone());
		frame_index++;
	}

	m_TotalFrames = frame_index;

	genCutFrames(m_FrameFilePath, cuts, allFrames);
}

void EdgeChangeRatio::algo()
{
	std::filesystem::remove_all(m_FrameFilePath);

	cv::Mat frame, grayFrame, blurFrame, sobelFrame, dilateFrame, invFrame, prevSobelFrame, prevInvFrame;
	uint32_t prevEdgePixels;

	uint frame_index = 1;

	while (true) {
		if (!getVideo().read(frame)) break;

		cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

		cv::GaussianBlur(grayFrame, blurFrame, cv::Size(3, 3), 0);

		cv::Sobel(blurFrame, sobelFrame, CV_64F, 0, 1, 5);

		uint32_t currEdgePixels = 0;

		uint8_t* imageData = (uint8_t*)sobelFrame.data;
		for (int i = 0; i < sobelFrame.rows; i++) {
			for (int j = 0; j < sobelFrame.cols; j++) {
				if(imageData[i * sobelFrame.cols + j]!=0) currEdgePixels++;
			}
		}

		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

		cv::dilate(sobelFrame, dilateFrame, kernel);

		cv::bitwise_not(dilateFrame, invFrame);

		if (frame_index != 1) {
			cv::Mat currFinalFrame, prevFinalFrame;
			cv::bitwise_and(prevInvFrame, sobelFrame, currFinalFrame);
			cv::bitwise_and(invFrame, prevSobelFrame, prevFinalFrame);

			uint8_t* prevData = (uint8_t*)prevFinalFrame.data;
			uint8_t* currData = (uint8_t*)currFinalFrame.data;
			
			uint32_t prevEdgeChange = 0;
			uint32_t currEdgeChange = 0;
			for (int i = 0; i < prevFinalFrame.rows; i++) {
				for (int j = 0; j < prevFinalFrame.cols; j++) {
					if (prevData[i * prevFinalFrame.cols + j] != 0) prevEdgeChange++;
					if (currData[i * currFinalFrame.cols + j] != 0) currEdgeChange++;
				}
			}

			double value = std::max(prevEdgePixels / prevEdgeChange, currEdgePixels / currEdgeChange);
			m_ResultFile << frame_index << "\t\t\t" << value << '\n';

		}

		prevInvFrame = invFrame.clone();
		prevSobelFrame = sobelFrame.clone();
		prevEdgePixels = currEdgePixels;

		frame_index++;
	}
	m_TotalFrames = frame_index;
}

void MutualInfo_Cut::algo()
{
	//cv::Mat frame, grayFrame;
	//uint8_t* prevIntensityB, *prevIntensityG, *prevIntensityR;

	//// First Frame
	//getVideo().read(frame);
	//prevIntensityB = new uint8_t[grayFrame.rows * grayFrame.cols];
	//prevIntensityG = new uint8_t[grayFrame.rows * grayFrame.cols];
	//prevIntensityR = new uint8_t[grayFrame.rows * grayFrame.cols];
	//
	//int channels = frame.channels();
	//uint8_t* imageData = (uint8_t*)frame.data;
	//for (int i = 0; i < frame.rows; i++) {
	//	for (int j = 0; j < frame.cols; j++) {
	//		prevIntensityB[i * frame.cols + j] = imageData[i*frame.cols*channels + j * channels + 0];
	//		prevIntensityG[i * frame.cols + j] = imageData[i*frame.cols*channels + j * channels + 1];
	//		prevIntensityR[i * frame.cols + j] = imageData[i*frame.cols*channels + j * channels + 2];
	//	}
	//}

	//uint frame_index = 1;
	//while (true) {
	//	if (!getVideo().read(frame)) break;

	//	uint8_t* newIntensityB, *newIntensityG, *prevIntensityR;

	//	prevIntensityB = new uint8_t[grayFrame.rows * grayFrame.cols];
	//	prevIntensityG = new uint8_t[grayFrame.rows * grayFrame.cols];
	//	prevIntensityR = new uint8_t[grayFrame.rows * grayFrame.cols];

	//	int channels = frame.channels();
	//	uint8_t* imageData = (uint8_t*)frame.data;
	//	for (int i = 0; i < frame.rows; i++) {
	//		for (int j = 0; j < frame.cols; j++) {
	//			prevIntensityB[i * frame.cols + j] = imageData[i*frame.cols*channels + j * channels + 0];
	//			prevIntensityG[i * frame.cols + j] = imageData[i*frame.cols*channels + j * channels + 1];
	//			prevIntensityR[i * frame.cols + j] = imageData[i*frame.cols*channels + j * channels + 2];
	//		}
	//	}

	//	double dIntensity = 0;

	//	if (frame_index != 1)
	//		m_ResultFile << frame_index << "\t\t\t" << dIntensity / (frame.rows * frame.cols) << '\n';
	//	frame_index++;
	//}

	//m_TotalFrames = frame_index;
	//
	//delete prevIntensityB, prevIntensityG, prevIntensityR;
}


