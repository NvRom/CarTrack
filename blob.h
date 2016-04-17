// Blob.h

#ifndef MY_BLOB
#define MY_BLOB

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

/************************************************************************/
/* 块，被追踪的物体用矩形块包围着		                                */
/************************************************************************/
class Blob {
public:
	// member variables ///////////////////////////////////////////////////////////////////////////
	std::vector<cv::Point> currentContour;

	//块面积
	cv::Rect currentBoundingRect;

	//块中心
	std::vector<cv::Point> centerPositions;
	cv::Point predictedNextPosition;

	//块的大小
	double dblCurrentDiagonalSize;
	double dblCurrentAspectRatio;

	//是否匹配
	bool blnCurrentMatchFoundOrNewBlob;

	//是否正在被追踪
	bool blnStillBeingTracked;
	//是否被追踪过，但现在没追踪
	bool blnHaveBeenCrossingLine;

	//
	int intNumOfConsecutiveFramesWithoutAMatch;
	//被追踪的块的编号
	int intNumOfTrackedCar;

	// function prototypes ////////////////////////////////////////////////////////////////////////
	Blob(std::vector<cv::Point> _contour);
	void predictNextPosition(void);

};
template<typename inputType>
inputType round(inputType i);
#endif    // MY_BLOB
