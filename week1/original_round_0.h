//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League code test
//////////////////////////////////////////////////////////////////////////
#ifndef __Round_0_h__
#define __Round_0_h__

#include <chrono>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="Team_X";
	std::string author_1="Name_1";
	std::string author_2="Name_2";
	std::string author_3="Name_3";	////optional
};

//////////////////////////////////////////////////////////////////////////
////TODO 1: Please replace the following code to calculate the sum of an integer array 
////with your own implementation to get better performance
//////////////////////////////////////////////////////////////////////////

int Int_Vector_Sum(const int* array,const int size)
{
	int sum=0;
	for(int i=0;i<size;i+=1){
		sum+=array[i];
	}
	return sum;
}

//////////////////////////////////////////////////////////////////////////
////TODO 2: Please replace the following code to calculate the sum of a double array 
////with your own implementation to get better performance
//////////////////////////////////////////////////////////////////////////

double Double_Vector_Sum(const double* array,const int size)
{
	double sum=0;
	for(int i=0;i<size;i++){
		sum+=array[i];
	}
	return sum;
}

#endif