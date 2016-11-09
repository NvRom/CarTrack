// Blob.h

#ifndef MY_BLOB
#define MY_BLOB

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

/************************************************************************/
/* �飬��׷�ٵ������þ��ο��Χ��		                                */
/************************************************************************/
class Blob {
public:
	// member variables ///////////////////////////////////////////////////////////////////////////
	std::vector<cv::Point> currentContour;

	//�����
	cv::Rect currentBoundingRect;

	//������
	std::vector<cv::Point> centerPositions;
	cv::Point predictedNextPosition;

	//��Ĵ�С
	double dblCurrentDiagonalSize;
	double dblCurrentAspectRatio;

	//�Ƿ�ƥ��
	bool blnCurrentMatchFoundOrNewBlob;

	//�Ƿ����ڱ�׷��
	bool blnStillBeingTracked;
	//�Ƿ�׷�ٹ���������û׷��
	bool blnHaveBeenCrossingLine;

	//
	int intNumOfConsecutiveFramesWithoutAMatch;
	//��׷�ٵĿ�ı��
	int intNumOfTrackedCar;

	// function prototypes ////////////////////////////////////////////////////////////////////////
	Blob(std::vector<cv::Point> _contour);
	void predictNextPosition(void);

};
template<typename inputType>
inputType round(inputType i);
#endif    // MY_BLOB
