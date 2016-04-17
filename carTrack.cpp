// carTrack.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h> 

#include "Blob.h"


const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

enum CAR_DIRECTION{BOTTOM_UP , TOP_DOWN};
int direction = BOTTOM_UP;
//��׷�ٵĳ�����Ŀ
int intNumTrackedCar = 0;

// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    std::vector<Blob> blobs;

    cv::Point crossingLine[2];

    int carCount = 0;

    capVideo.open("CarsDrivingUnderBridge.mp4");
	//����Ƶû�ɹ��򿪻���ֻ��һ֡ʱ���˳�
    if (!capVideo.isOpened()) {
        std::cout << "error reading video file" << std::endl << std::endl;      
        return(0);
    }

    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "error: video file must have at least two frames";
        return(0);
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

	//crossing��
    int intHorizontalLinePosition = (int)round((double)imgFrame1.rows * 0.35);
	int intVerticalLinePosition = 0;

    crossingLine[0].x = intVerticalLinePosition;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = imgFrame1.cols - 1;
    crossingLine[1].y = intHorizontalLinePosition;

    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;

    while (capVideo.isOpened() && chCheckForEscKey != 27) {
/************************************************************************/
/*							ǰ����ȡ                                    */
/************************************************************************/
        std::vector<Blob> currentFrameBlobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        cv::Mat imgDifference;
        cv::Mat imgThresh;
		//The function converts an input image from one color space to another
		//point��http://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
		/*
		http://docs.opencv.org/3.1.0/de/db2/laplace_8cpp-example.html#a13
		*/
        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);
		/*Calculates the per-element absolute difference between two arrays or between an array and a scalar*/
        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
		/*Applies a fixed-level threshold to each array element
		The function applies fixed-level thresholding to a single-channel array
		*/
        cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);
		//cv::adaptiveThreshold();

		//��̬ѧ����
        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        for (unsigned int i = 0; i < 2; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::erode(imgThresh, imgThresh, structuringElement5x5);
        }
/************************************************************************/
/*								��ת��                                  */
/************************************************************************/
        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


        std::vector<std::vector<cv::Point> > convexHulls(contours.size());
		/*
		http://docs.opencv.org/3.1.0/d7/d1d/tutorial_hull.html
		*/
        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

		/*http://docs.opencv.org/3.1.0/d7/d1d/tutorial_hull.html*/
        for (auto &convexHull : convexHulls) {
            Blob possibleBlob(convexHull);

            if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 30 &&
                possibleBlob.currentBoundingRect.height > 30 &&
                possibleBlob.dblCurrentDiagonalSize > 60.0 &&
				possibleBlob.currentBoundingRect.y + 100> intHorizontalLinePosition &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }
/************************************************************************/
/*								�����                                  */
/************************************************************************/

        if (blnFirstFrame == true) {
            for (auto &currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        } else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
        }


        imgFrame2Copy = imgFrame2.clone();
		//�þ��ο���������������
        drawBlobInfoOnImage(blobs, imgFrame2Copy);

        bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount);
		//��������ʱ������ɫ���;ƽʱΪ��ɫ
        if (blnAtLeastOneBlobCrossedTheLine == true) {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
        }
        else {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
        }
		//��������������Ŀ��ʾ����Ƶ��
        drawCarCountOnImage(carCount, imgFrame2Copy);

        cv::imshow("imgFrame2Copy", imgFrame2Copy);

        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);
        }
        else {
            std::cout << "end of video\n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
    }

    if (chCheckForEscKey != 27) {
        cv::waitKey(0);
	}

    return(0);
}

/*----------------------------------------------------------------------*/
/*						���ǻ������ķָ���								*/
/*						  �����Ǻ�������								*/
/*----------------------------------------------------------------------*/

/************************************************************************/
/*			�˺����������ǵ�ǰframe�еĿ����Ѵ��ڵĿ����ƥ��			*/
/*���������															*/
/*vector<Blob> &existingBlobs���Ѵ��ڵĿ鼯��							*/
/*vector<Blob> &currentFrameBlobs����ǰframe�Ŀ鼯��					*/
/************************************************************************/
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

    for (auto &existingBlob : existingBlobs) {

        existingBlob.blnCurrentMatchFoundOrNewBlob = false;

        existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;
		//Ѱ������֪����ӽ��Ŀ飬�������±�
        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

			if (existingBlobs[i].blnStillBeingTracked == true) {

                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
			else if(existingBlobs[i].centerPositions.size() == 1){
				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
			}
        }
		//�����޸�dblCurrentDiagonalSize��ϵ����0.5���ҵľ���ֵ
        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }

    }
//��ȥһЩǰһ֡���ں�һ֡�����ڵĳ�����������ʧ�ĳ�
    for (auto &existingBlob : existingBlobs) {

        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;

        }

    }

}
/************************************************************************/
/*                           �����Ѵ��ڵĿ�                             */
/*��ԭ���е�currentContour��currentBoundingRect�������޸�				*/
/*ͬʱ��vector<point>centerPositions���һ���µ�centerPositions			*/
/************************************************************************/
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {
	//����Ҫ��⳵�ķ���ѡ���޸�existingBlobs
	if (direction == BOTTOM_UP && currentFrameBlob.currentBoundingRect.y< existingBlobs[intIndex].currentBoundingRect.y)
	{
		//��׽�����³�����
		if(existingBlobs[intIndex].centerPositions.size() == 2 && 
			existingBlobs[intIndex].blnStillBeingTracked == true && 
			existingBlobs[intIndex].blnHaveBeenCrossingLine == false){
			existingBlobs[intIndex].intNumOfTrackedCar = ++ intNumTrackedCar;
		}
		//�޸�blob�е�һЩֵ
		existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
		existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

		existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

		existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
		existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

		existingBlobs[intIndex].blnStillBeingTracked = true;
		existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
	}
    
}

/************************************************************************/
/*								����¿�								*/
/************************************************************************/
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs.push_back(currentFrameBlob);
}

/************************************************************************/
/*							�����ľ���                                */
/************************************************************************/
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {
    
    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

/************************************************************************/
/*						�ж��Ƿ񾭹�������������                        */
/*	������centerPositions�ֱ���ˮƽ�߽��бȽϣ�һ��һС��˵��ǡ�þ���	*/
/************************************************************************/
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
    bool blnAtLeastOneBlobCrossedTheLine = false;

    for (unsigned int i = 0 ; i < blobs.size() ; i ++) {

        if (blobs[i].blnStillBeingTracked == true && blobs[i].centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blobs[i].centerPositions.size() - 2;
            int currFrameIndex = (int)blobs[i].centerPositions.size() - 1;

            if (blobs[i].centerPositions[prevFrameIndex].y > intHorizontalLinePosition && 
				blobs[i].centerPositions[currFrameIndex].y <= intHorizontalLinePosition){
                carCount++;
                blnAtLeastOneBlobCrossedTheLine = true;
				//ײ��֮��Ͳ��ٲ�׽
				blobs[i].blnHaveBeenCrossingLine = true;
            }
        }

    }

    return blnAtLeastOneBlobCrossedTheLine;
}

/************************************************************************/
/*								���ƾ��ο�                              */
/************************************************************************/
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {
	//������ʾ��׷�ٹ��ĳ����ĸ���
    for (unsigned int i = 0; i < blobs.size(); i++) {

		if (blobs[i].blnStillBeingTracked == true && blobs[i].blnHaveBeenCrossingLine == false) {

            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);
            int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
            double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
            int intFontThickness = (int)round(dblFontScale * 1.0);
			if(blobs[i].intNumOfTrackedCar != 0){
				cv::putText(imgFrame2Copy, std::to_string(blobs[i].intNumOfTrackedCar), 
					blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
        	}
		}
    }
}

/************************************************************************/
/*							��Ӿ���������Ŀ                            */
/************************************************************************/
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
    int intFontThickness = (int)round(dblFontScale * 1.5);

    cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

    cv::Point ptTextBottomLeftPosition;

    ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
    ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

    cv::putText(imgFrame2Copy, std::to_string(carCount), 
		ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

}
