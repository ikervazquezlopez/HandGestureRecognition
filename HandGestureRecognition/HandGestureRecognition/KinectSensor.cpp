#include "stdafx.h"
#include "KinectSensor.h"


KinectSensor::KinectSensor()
{
	HRESULT hr;


	hr = GetDefaultKinectSensor(&pKinectSensor);
	if (FAILED(hr))
		error("ERROR : getDefaultKinectSensor() : ", hr);

	hr = pKinectSensor->Open();
	if (FAILED(hr))
		error("ERROR : pKinectSensor->open() : ", hr);



	// GET THE DEPTH READER //
	IDepthFrameSource * pDepthFrameSource;
	hr = pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
	if (FAILED(hr))
		error("ERROR : pKinectSensor->getDepthFrameSource() : ", hr);
	hr = pDepthFrameSource->OpenReader(&pDepthFrameReader);
	if (FAILED(hr))
		error("ERROR : pDepthFrameSource->OpenReader() : ", hr);
	SafeRelease<IDepthFrameSource>(pDepthFrameSource);



	// GET THE COLOR READER //
	IColorFrameSource * pColorFrameSource;
	hr = pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
	if (FAILED(hr))
		error("ERROR : pKinectSensor->getColorFrameSource() : ", hr);
	hr = pColorFrameSource->OpenReader(&pColorFrameReader);
	if (FAILED(hr))
		error("ERROR : pColorFrameSource->OpenReader() : ", hr);
	SafeRelease<IColorFrameSource>(pColorFrameSource);



	// GET THE INFRARED READER //
	IInfraredFrameSource * pInfraredFrameSource;
	hr = pKinectSensor->get_InfraredFrameSource(&pInfraredFrameSource);
	if (FAILED(hr))
		error("ERROR : pKinectSensor->getInfraredFrameSource() : ", hr);
	hr = pInfraredFrameSource->OpenReader(&pInfraredFrameReader);
	if (FAILED(hr))
		error("ERROR : pInfraredFrameSource->OpenReader() : ", hr);
	SafeRelease<IInfraredFrameSource>(pInfraredFrameSource);

	// GET THE BODY READER //
	IBodyFrameSource * pBodyFrameSource;
	hr = pKinectSensor->get_BodyFrameSource(&pBodyFrameSource);
	if (FAILED(hr))
		error("ERROR : pKinectSensor->getBodyFrameSource() : ", hr);
	hr = pBodyFrameSource->OpenReader(&pBodyFrameReader);
	if (FAILED(hr))
		error("ERROR : pBodyFrameSource->OpenReader() : ", hr);
	SafeRelease<IBodyFrameSource>(pBodyFrameSource);

	hr = pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
	if (FAILED(hr))
		error("ERROR : pKinectSensor->getCoordinateMapper() : ", hr);
}


KinectSensor::~KinectSensor()
{
}

IDepthFrame * KinectSensor::getNextDepthFrame(){
	IDepthFrame * depthFrame = NULL;
	HRESULT hr = pDepthFrameReader->AcquireLatestFrame(&depthFrame);
	if (FAILED(hr))
		error("ERROR : pDepthFrameReader->AcquireLatestFrame() : ", hr);
	return depthFrame;
}

IColorFrame * KinectSensor::getNextColorFrame(){
	IColorFrame * colorFrame = NULL;
	HRESULT hr = pColorFrameReader->AcquireLatestFrame(&colorFrame);
	if (FAILED(hr))
		error("ERROR : pColorFramReader->AcquireLatestFrame() : ", hr);
	return colorFrame;
}

IInfraredFrame * KinectSensor::getNextInfraredFrame(){
	IInfraredFrame * infraredFrame = NULL;
	HRESULT hr = pInfraredFrameReader->AcquireLatestFrame(&infraredFrame);
	if (FAILED(hr))
		error("ERROR : pInfraredFrameReader->AcquireLatestFrame() : ", hr);
	return infraredFrame;
}

IBodyFrame * KinectSensor::getNextBodyFrame(){
	IBodyFrame * bodyFrame = NULL;

	HRESULT hr = pBodyFrameReader->AcquireLatestFrame(&bodyFrame);
	if (FAILED(hr))
		error("ERROR : pBodyFrameReader->AcquireLatestFrame() : ", hr);
	return bodyFrame;
}



cv::Mat convertRawToMat(UINT16 *inputArray, int rows, int cols, int outputFormat){
	return cv::Mat::zeros(1, 1, CV_16U);
}


IDepthFrameReader * KinectSensor::getDepthFrameReader(){
	return pDepthFrameReader;
}


IColorFrameReader * KinectSensor::getColorFrameReader(){
	return pColorFrameReader;
}


IInfraredFrameReader * KinectSensor::getInfraredFrameReader(){
	return pInfraredFrameReader;
}

IBodyFrameReader * KinectSensor::getBodyFrameReader(){
	return pBodyFrameReader;
}

ICoordinateMapper * KinectSensor::getCoordinateMapper(){
	return m_pCoordinateMapper;
}

template<class Interface>
void KinectSensor::SafeRelease(Interface *& pInterface){
	if (pInterface != NULL){
		pInterface->Release();
		pInterface = NULL;
	}
}


void KinectSensor::error(std::string e, HRESULT hr){
	std::cout << e;
	std::cout << hr;
	Sleep(5000);
	exit(-1);
}