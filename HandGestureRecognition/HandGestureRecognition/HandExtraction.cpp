#include "stdafx.h"
#include "HandExtraction.h"




cv::Mat extract_hand_region(IDepthFrame * depthFrame, Joint handJoint, bool bTracked){
	HRESULT hr;

	/* Frame description*/
	IFrameDescription * frameDesc; depthFrame->get_FrameDescription(&frameDesc);
	int height, width;
	frameDesc->get_Height(&height); frameDesc->get_Width(&width);

	cv::Mat crop = cv::Mat::zeros(2 * HAND_RECTANGLE_WIDTH, 2 * HAND_RECTANGLE_HEIGTH, CV_16UC3);
	cv::Mat extracted_hand = cv::Mat::zeros(height, width, CV_16UC3);
	int frame_size = width * height;

	/* Get hand coordinates */
	float x, y, z;
	x = (handJoint.Position.X + 1) * width / 2 + HAND_DISPLACEMENT_CORRECTION;
	y = (handJoint.Position.Y + 1) * height / 2 + HAND_DISPLACEMENT_CORRECTION; y = height - y;
	z = handJoint.Position.Z * 1000;


	UINT16 * pBuffer = NULL;
	pBuffer = new UINT16[frame_size];
	hr = depthFrame->CopyFrameDataToArray(frame_size, pBuffer);


	if (SUCCEEDED(hr)){
		if (bTracked){
			// Compute depth threshold //
			for (int i = 0; i < frame_size; i++){ //Change the position of this to get better performance (after cropping the image)
				int x_index_coord = i % width;
				int y_index_coord = i / width;

				if (x_index_coord > x - HAND_RECTANGLE_WIDTH && x_index_coord < x + HAND_RECTANGLE_WIDTH  &&
					y_index_coord > y - HAND_RECTANGLE_HEIGTH && y_index_coord < y + HAND_RECTANGLE_HEIGTH)
				{
					if (pBuffer[i] > z + HAND_EXTRACTCION_DEPTH_THRESHOLD ||
						pBuffer[i] < z - HAND_EXTRACTCION_DEPTH_THRESHOLD)
					{
						pBuffer[i] = 0;
					}
				}
				else {
					pBuffer[i] = 0;
				}
			}

			/* Create cropping rectangle */
			int x_min = (int)floor(x - HAND_RECTANGLE_WIDTH);
			int y_min = (int)floor(y - HAND_RECTANGLE_HEIGTH);

			if (x_min < 0){ x_min = 0; }
			if (y_min < 0){ y_min = 0; }
			if (x_min + 2 * HAND_RECTANGLE_WIDTH > width){ x_min = width - 2 * HAND_RECTANGLE_WIDTH; }
			if (y_min + 2 * HAND_RECTANGLE_HEIGTH > height){ y_min = height - 2 * HAND_RECTANGLE_HEIGTH; }

			cv::Rect croppingRect(x_min, y_min, 2 * HAND_RECTANGLE_WIDTH, 2 * HAND_RECTANGLE_HEIGTH);

			/* Scale the data */
			USHORT minDepth; depthFrame->get_DepthMinReliableDistance(&minDepth);
			USHORT maxDepth; depthFrame->get_DepthMaxReliableDistance(&maxDepth);
			double scale = 1.0; // double scale = 255.0 / (maxDepth - minDepth);

			/* Convert and crop the image */
			cv::Mat depthMap = cv::Mat(height, width, CV_16U, pBuffer);
			depthMap.convertTo(extracted_hand, CV_16UC3, scale);
			crop = extracted_hand(croppingRect);

			free(pBuffer);
			depthMap.release();
		}

	}
	return crop;
}


cv::Mat extract_hand_region(IColorFrame * colorFrame, Joint handJoint, bool bTracked){
	HRESULT hr;
	BYTE * pBuffer = NULL;
	cv::Mat trueColor, color;

	/* Frame description*/
	IFrameDescription * frameDesc; colorFrame->get_FrameDescription(&frameDesc);
	int height, width;
	frameDesc->get_Height(&height); frameDesc->get_Width(&width);

	cv::Mat crop = cv::Mat::zeros(2 * HAND_RECTANGLE_WIDTH, 2 * HAND_RECTANGLE_HEIGTH, CV_8UC3);
	cv::Mat extracted_hand = cv::Mat::zeros(height, width, CV_8UC3);
	int frame_size = width * height;

	/* Get hand coordinates */
	float x, y, z;
	x = (handJoint.Position.X + 1) * KINECT_DEPTH_WIDTH / 2;
	y = (handJoint.Position.Y + 1) * KINECT_DEPTH_HEIGTH / 2; y = KINECT_DEPTH_HEIGTH - y;
	z = handJoint.Position.Z * 1000;

	// Convert the Frame to an array //
	pBuffer = new BYTE[frame_size * sizeof(RGBQUAD)];
	hr = colorFrame->CopyConvertedFrameDataToArray(frame_size * sizeof(RGBQUAD),
		pBuffer, ColorImageFormat_Rgba);

	if (SUCCEEDED(hr)){
		trueColor = cv::Mat(height, width, CV_8UC4,
			reinterpret_cast<void*>(pBuffer));

		cv::resize(trueColor, color, cv::Size(KINECT_DEPTH_WIDTH, KINECT_DEPTH_HEIGTH));

		width = KINECT_DEPTH_WIDTH;
		height = KINECT_DEPTH_WIDTH;

		/* Create cropping rectangle */
		int x_min = (int)floor(x - HAND_RECTANGLE_WIDTH);
		int y_min = (int)floor(y - HAND_RECTANGLE_HEIGTH);

		if (x_min < 0){ x_min = 0; }
		if (y_min < 0){ y_min = 0; }
		if (x_min + 2 * HAND_RECTANGLE_WIDTH > width){ x_min = width - 2 * HAND_RECTANGLE_WIDTH; }
		if (y_min + 2 * HAND_RECTANGLE_HEIGTH > height){ y_min = height - 2 * HAND_RECTANGLE_HEIGTH; }

		//cv::Rect croppingRect(x_min, y_min, 2 * HAND_RECTANGLE_WIDTH, 2 * HAND_RECTANGLE_HEIGTH);
		cv::Rect croppingRect(x_min, y_min, 2 * HAND_RECTANGLE_WIDTH, 2 * HAND_RECTANGLE_HEIGTH);


		/* Convert and crop the image */
		color.convertTo(extracted_hand, CV_8UC3);
		crop = extracted_hand(croppingRect);

		free(pBuffer);
		color.release();
		trueColor.release();
		extracted_hand.release();
	}
	//extracted_hand = cv::Mat::zeros(100, 100, CV_8UC3);
	return crop;
}


cv::Mat extract_borders(cv::Mat input, int threshold1, int threshold2){
	cv::Mat borders, blurred;

	/* Extract borders*/
	cv::blur(input, blurred, cv::Size2d(3, 3), cv::Point(0, 0));
	cv::Canny(blurred, borders, threshold1, threshold2);

	/* Free variables */
	blurred.release();
	return borders;
}



cv::Mat getFourierTransformation(cv::Mat input){
	cv::Mat padded;                            //expand input image to optimal size
	int m = cv::getOptimalDFTSize(input.rows);
	int n = cv::getOptimalDFTSize(input.cols); // on the border add zero values
	copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	cv::Mat magI = planes[0];

	magI += cv::Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	cv::normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

	return input;
}