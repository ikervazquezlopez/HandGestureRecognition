#include "stdafx.h"

#define HAND_EXTRACTCION_DEPTH_THRESHOLD	60		//In milimeters (Kinect maximun range is 8000->8m)
#define HAND_RECTANGLE_WIDTH				50
#define HAND_RECTANGLE_HEIGTH				50
#define HAND_DISPLACEMENT_CORRECTION		15		//It is arbitrary at this moment

#define KINECT_DEPTH_WIDTH 512
#define KINECT_DEPTH_HEIGTH 424


/* Extracts the sourronding region of the hand in x, y and z(the depth) from a depth image */
cv::Mat extract_hand_region(IDepthFrame * depthFrame, Joint handJoint, bool bTracked);

/* Extracts the sourronding region of the hand in x and y from a color image */
cv::Mat extract_hand_region(IColorFrame * colorFrame, Joint handJoint, bool bTracked);

/* Extract the color region from the depth data in depthImage */
cv::Mat extract_colorHand_from_depthHand(cv::Mat colorImage, cv::Mat depthImage);

/* Detects the borders of the image */
cv::Mat extract_borders(cv::Mat input, int threshold1, int threshold2);

/* Gets the fourier transformation of the image */
cv::Mat getFourierTransformation(cv::Mat input);