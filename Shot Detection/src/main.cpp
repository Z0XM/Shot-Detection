#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "Algorithms.hpp"

void GNUPlotSingle(Algorithm* algorithm) {
	FILE* gnupipe = _popen("./gnuplot/bin/gnuplot.exe -persistent", "w");
	if (gnupipe == NULL) {
		std::cout << "Error Opening gnuplot\n";
		return;
	}

	fprintf(gnupipe, "set xlabel '%s'\n", algorithm->getXlabel().c_str());
	fprintf(gnupipe, "set ylabel '%s'\n", algorithm->getYlabel().c_str());
	fprintf(gnupipe, "set xrange [2:%d]\n", algorithm->getTotalFrames());
	fprintf(gnupipe, "plot '%s' title '%s' lt 7 lc 24 w lp\n", algorithm->getResultFilePath().c_str(), algorithm->getTitle().c_str());
}

void GNUPlotMultiple(const std::vector<Algorithm*>& algorithms)
{
	if (algorithms.empty()) return;

	FILE* gnupipe = _popen("gnuplot -persistent", "w");
	if (gnupipe == NULL) {
		std::cout << "Error Opening gnuplot\n";
		return;
	}

	fprintf(gnupipe, "set xlabel '%s'\n", algorithms[0]->getXlabel().c_str());
	fprintf(gnupipe, "set ylabel '%s'\n", algorithms[0]->getYlabel().c_str());
	fprintf(gnupipe, "set xrange [2:%d]\n", algorithms[0]->getTotalFrames());


	std::string cmd = "plot ";
	int color = 1;
	for (Algorithm* algo : algorithms) {
		cmd += "'" + algo->getResultFilePath() + "' title '" + algo->getTitle() + "' lt 7 lc " + std::to_string(color) + " w lp, ";
		color++;
	}

	fprintf(gnupipe, "%s\n", cmd.c_str());
}

//int main()
//{
//	Algorithm* algorithm = new PixelDifferenceColor("./res/StrangeClip.mp4");
//
//	if (!algorithm->getError())
//		algorithm->run();
//
//	if (!algorithm->getError())
//		GNUPlotSingle(algorithm);
//
//	delete algorithm;
//}

int main()
{
	char filePath[] = "./res/Strange12.mp4";
	std::vector<Algorithm*> algorithms;
	//algorithms.push_back(new PixelDifference(filePath));
	//algorithms.push_back(new PixelDifferenceColor(filePath));
	//algorithms.push_back(new Histogram_Bin2Bin(filePath));
	//algorithms.push_back(new Histogram_ChiSqrOld(filePath));
	//algorithms.push_back(new Histogram_ChiSqrNew(filePath));
	//algorithms.push_back(new Histogram_Intersect(filePath));

	bool error = false;
	int i = 0;
	for (Algorithm* algo : algorithms) {
		std::cout << "Algo : " << i << '\n';
		algo->run();
		if (error = algo->getError()) break;
		i++;
	}

	if (!error) {
		GNUPlotMultiple(algorithms);
	}


	for (Algorithm* algo : algorithms) {
		delete algo;
	}

}