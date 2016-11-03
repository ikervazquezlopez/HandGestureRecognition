#include <opencv2\core.hpp>






#include "utilites.h"


std::string getClassFromFilename(std::string filename){
	char* name = (char *) filename.c_str();
	char * imageClass = std::strtok(name, " ");
	return imageClass;
}