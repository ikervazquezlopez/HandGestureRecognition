#pragma once


#include "stdafx.h"

#define FRAME_DEPTH		0
#define FRAME_COLOR		1
#define FRAME_INFRARED	2
#define FRAME_BODY		3


class KinectSensor
{

private:
	IKinectSensor * pKinectSensor;

	IDepthFrameReader * pDepthFrameReader;
	IColorFrameReader * pColorFrameReader;
	IInfraredFrameReader * pInfraredFrameReader;
	IBodyFrameReader * pBodyFrameReader;

	ICoordinateMapper * m_pCoordinateMapper;

	void error(std::string e, HRESULT hr);

public:
	KinectSensor();
	~KinectSensor();

	IDepthFrameReader * getDepthFrameReader();
	IColorFrameReader * getColorFrameReader();
	IInfraredFrameReader * getInfraredFrameReader();
	IBodyFrameReader * getBodyFrameReader();
	ICoordinateMapper * getCoordinateMapper();


	/* Returns the next frame from the depth reader */
	IDepthFrame * getNextDepthFrame();

	/* Returns the next frame from the color reader */
	IColorFrame * getNextColorFrame();

	/* Returns the next frame from the depth reader */
	IInfraredFrame * getNextInfraredFrame();

	/* Returns the next frame from the depth reader */
	IBodyFrame * getNextBodyFrame();

	/* Converts Kinect raw type to OpenCV Mat type */
	template<typename Type>
	cv::Mat convertRawToMat(Type *inputArray, int rows, int cols, int inputFormat, int outputFormat);//TODO

	/* Extracts a rectangle formed by the points p0 and p1 */
	cv::Mat zoneExtraction(cv::Mat input, cv::Point2d p0, cv::Point2d p1);//TODO

	/* Returns the tracked body joints in a frame */
	std::vector<Joint> frameToBodyJoints(IDepthFrame * inputFrame); //TODO

	/* Releases any Kinect interface */
	template<class Interface>
	void SafeRelease(Interface *& pInterface);//TODO

};

