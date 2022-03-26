#pragma once
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <string>

class Algorithm{
public:
	Algorithm(const std::string& videoFilepath, const std::string& title, const std::string& ylabel, const std::string& xlabel, uint32_t threshold)
		:m_Video(videoFilepath),
		m_ResultFilePath("./results/data/" + title + ".dat"),
		m_Title(title), m_Ylabel(ylabel), m_Xlabel(xlabel),
		m_FrameFilePath("./results/frames/" + title),
		m_TotalFrames(0), m_Error(false), m_Threshold(threshold)
	{
		if (!m_Video.isOpened()) {
			std::cout << "Error opening video stream\n";
			m_Error = true;
		}
	}

	inline cv::VideoCapture& getVideo() { return m_Video; }
	inline bool getError() { return m_Error; }
	inline uint getTotalFrames() { return m_TotalFrames; }
	inline const std::string& getResultFilePath() const { return m_ResultFilePath; }
	inline const std::string& getTitle() const { return m_Title; }
	inline const std::string& getYlabel() const { return m_Ylabel; }
	inline const std::string& getXlabel() const { return m_Xlabel; }

	void run() {
		m_ResultFile.open(getResultFilePath().c_str());
		if (!m_ResultFile.is_open()) {
			std::cout << "Unable to Open File " << getResultFilePath() << '\n';
			m_Error = true;
			return;
		}

		algo();

		m_ResultFile.close();
	};

	virtual void algo() = 0;

private:
	cv::VideoCapture m_Video;
	std::string m_ResultFilePath, m_Title, m_Ylabel, m_Xlabel;

protected:
	std::string m_FrameFilePath;
	uint32_t m_TotalFrames;
	bool m_Error;
	uint32_t m_Threshold;
	std::ofstream m_ResultFile;
};

class PixelDifference : public Algorithm {
public:
	PixelDifference(const char* filepath = 0)
		:Algorithm(filepath, "Pixel Difference", "Intensity Difference", "Frame Index", 1e7)
	{
	}

	void algo() override;
};

class PixelDifferenceColor : public Algorithm {
public:
	PixelDifferenceColor(const char* filepath = 0)
		:Algorithm(filepath, "Pixel Difference Color", "Intensity Difference", "Frame Index", 3e7)
	{
	}

	void algo() override;
};

class Histogram_Bin2Bin : public Algorithm {
public:
	Histogram_Bin2Bin(const char* filepath = 0)
		:Algorithm(filepath, "Historgram Bin-to-Bin", "Histogram Difference", "Frame Index", 2)
	{
	}

	void algo() override;
};

class Histogram_ChiSqrNew : public Algorithm {
public:
	Histogram_ChiSqrNew(const char* filepath = 0)
		:Algorithm(filepath, "Historgram Chi Square New", "Histogram Difference", "Frame Index", 2)
	{
	}

	void algo() override;
};

class Histogram_ChiSqrOld : public Algorithm {
public:
	Histogram_ChiSqrOld(const char* filepath = 0)
		:Algorithm(filepath, "Historgram Chi Square Old", "Histogram Difference", "Frame Index", 2e4)
	{
	}

	void algo() override;
};


class Histogram_Intersect : public Algorithm {
public:
	Histogram_Intersect(const char* filepath = 0)
		:Algorithm(filepath, "Historgram Intersect", "Histogram Difference", "Frame Index", 5e8)
	{
	}

	void algo() override;
};


class MutualInfo_Cut : public Algorithm {
public:
	MutualInfo_Cut(const char* filepath = 0)
		:Algorithm(filepath, "Mutual Information Cut", "Mutual Information", "Frame Index", 0)
	{
	}

	void algo() override;
};
