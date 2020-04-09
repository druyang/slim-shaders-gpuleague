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
	std::string team="slim_shaders";
	std::string author_1="Andrw_Yang";
	std::string author_2="Matthew_Kenney";
};

//////////////////////////////////////////////////////////////////////////
////TODO 1: Please replace the following code to calculate the sum of an integer array 
////with your own implementation to get better performance
//////////////////////////////////////////////////////////////////////////

int Int_Vector_Sum(const int* array,const int size)
{

	int acc0 = 0;
	int acc1 = 0;
	int acc2 = 0;
	int acc3 = 0;
	int i;
	int limit = size - 1;
	
	for(i = 0; i < limit ; i += 4){
		acc0 += array[i];
		acc1 += array[i + 1];
		acc2 += array[i + 2];
		acc3 += array[i + 3];
	}

	// Finish up the split loop:
	for (; i < size; i++) {
		acc0 += array[i];
	}

	return acc0 + acc1 + acc2 + acc3;

}

//////////////////////////////////////////////////////////////////////////
////TODO 2: Please replace the following code to calculate the sum of a double array 
////with your own implementation to get better performance
//////////////////////////////////////////////////////////////////////////

double Double_Vector_Sum(const double* array,const int size)
{
	double sum=0;
	int i;
	int limit = size - 1;
	
	for(i = 0; i < limit ; i += 6){
		sum = sum + (array[i] + array[i+1] + array[i+2] + array[i+3] + array[i+4] + array[i+5]);
	}

	// Finish up the split loop:
	for (; i < size; i++) {
		sum += array[i];
	}

	return sum;
}

#endif
