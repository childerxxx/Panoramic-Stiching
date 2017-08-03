#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;
stringstream ss;
//#define MAX 10;

int seed =0, numimages=0;
Mat image[10];
//class imageOperations :  
Mat image_keypoints[10],seed_matches[10],output[10],imagegood_matches[10];
Mat image_descriptors[10];
vector<KeyPoint> vec_keypoints[10];
int w=0,h=0, x=0, y=0, i=0,j=0;
int numkp[10];	
int gmindex = 0,outputindex =0,smindex=0;
vector<Point2f> query;
vector<Point2f> train;
void computeAlignment(Mat image1, Mat image2,int first);

void detectFeatures(Mat img, int index)
{	
	int threshold = 400;
	//Ptr<FeatureDetector> detector1 = FeatureDetector::create("HARRIS");
	
	//Detecting features using Surf Detector
	SurfFeatureDetector detector(threshold);
	detector.detect( img, vec_keypoints[index]);
	drawKeypoints( img, vec_keypoints[index], image_keypoints[index], Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	
	//Using SurfExtractor to extract descriptors
	SurfDescriptorExtractor extractor;
	extractor.compute(img, vec_keypoints[index], image_descriptors[index]);
	
	//dynamically create filename --for testing
	numkp[j] = vec_keypoints[j].size();
	string kpfname = "image_keypoints";
	stringstream ss;
	ss<<index;
	kpfname = kpfname+ss.str()+".jpg";
	ss.clear();
	//imwrite(kpfname,image_keypoints[index]);
}

void match(Mat image1,Mat image2, int sd, int z)
{
	/*Matches 2 images and select good matches in them,orders images and call compute alignment*/
	
	double max_distance=0;
	double min_distance=100;
	query.clear();
	train.clear();
	/*Matching descriptor vectors using FLANN matcher */
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	vector<DMatch> good_matches;
	matcher.match(image_descriptors[sd], image_descriptors[z], matches);
	drawMatches(image1, vec_keypoints[sd], image2, vec_keypoints[z], matches, seed_matches[smindex]);
	
		string kpfname = "matches";
		stringstream ss;
		ss<<smindex;
		kpfname = kpfname+ss.str()+".jpg";
		ss.clear();
		imwrite(kpfname,seed_matches[smindex]);
	smindex++;
	
	//Finding minimuma nd maximum distances
	for(i=0;i<image_descriptors[sd].rows;i++)
	{
		double dist = matches[i].distance;
		if(dist<min_distance)
		{
		min_distance = dist;
		}
		if(dist>max_distance)
		{
		max_distance = dist;
		}
	}
	//cout<<"\nMaximum distance :"<<max_distance;
	//cout<<"\nMinimum distance :"<<min_distance;

	
	/* Using good points (whose distance is less than 2*min_distance)*/
	int num = 1;
	for(i=0;i<image_descriptors[sd].rows;i++)
	{
		if(matches[i].distance<max(2.5*min_distance,0.02))
		{
			num++;
			good_matches.push_back(matches[i]);
		}
		
	}
	
	/* Get keypoints from the good matches*/
	for(i=0;i<good_matches.size();i++)
	{
	query.push_back(vec_keypoints[sd][ good_matches[i].queryIdx].pt );
	train.push_back(vec_keypoints[z][ good_matches[i].trainIdx].pt );
	}
			//cout<<"\n matches:"<< matches.size();
			//cout<<"\n good_matches:"<< good_matches.size();
			//cout<<"\n train:"<< train.size();
			//cout<<"\n query:"<< query.size();
			//cout<<"\n query:"<< query.pt;
			//cout<<"\n matches:"<< matches.size();
	drawMatches( image1, vec_keypoints[sd], image2, vec_keypoints[z], good_matches, imagegood_matches[gmindex] );
	
	//dynamically create filename --for testing
	kpfname = "goodmatches";
	stringstream sk;
	sk<<gmindex;
	kpfname = kpfname+sk.str()+".jpg";
	sk.clear();
	imwrite(kpfname,imagegood_matches[gmindex]);
	gmindex++;
	
	/* Ordering images by counting number of good keypoints on left side of image and right side using counters lcount and rcount*/
	int min_x = image1.cols/2;
	int y = image1.rows;
	int first = 0;
	int lcount =0, rcount = 0;
	for(int i=0;i<query.size();i++)
	{
		Point qt = query.at(i);
		if(qt.x > min_x) rcount++;
		else lcount++;
	}
	if(lcount<=rcount) first=1;
	
	if(good_matches.size()<100)
	{
		if(first==0) computeAlignment(image2, image1,first); //warps the image and stiches them
		else computeAlignment(image1, image2,first);
	}
}
Mat cropImage(Mat crop)
{
	/*Finds minimum bounded rectangle and crops the stiches pair without data loss*/
	
	Mat canny_output = crop;
	RNG rng(12345);
	Mat contourRegion,subregions;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	int thresh=100;
	cvtColor( canny_output, canny_output, CV_BGR2GRAY );
    blur( canny_output, canny_output, Size(3,3) );
	//Canny( canny_output, canny_output, thresh, thresh*2, 3 );
	
	findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	int max_x = 0,max_y = 0,far[2],x=0,y=0,min_x=crop.cols,min_y=crop.rows;
	far[0]=0;
	far[1]=0;
	double maxArea = 0.0;
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
	}
	//cout<<"contours.size:"<<contours.size();
	vector<vector<Point> > contours_poly( contours.size() );
	Rect boundRect;
	for( int i = 0; i < contours.size(); i++ )
	{ 
		double area = contourArea(contours[i]);
		if(area>maxArea)
		{
			maxArea = area;
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect = boundingRect(Mat(contours_poly[i]));
				/*cout<<"contours_poly"<<contours_poly.size();
				cout<<"contours_poly"<<contours_poly[i];
					vector<Point> vp = contours_poly.at(j);
					for(int a = 0;a<vp.size();a++)
					{
						Point p = vp.at(a);
						x = p.x;
						y = p.y;
						//cout<<"x:"<<x<<"y:"<<y;
						if(far[0]*far[1]<x*y)
						{
							far[0] = x;
							far[1] = y;
						}
						if(max_x<x)
							max_x=x;
						else if(x!=1 && min_x >x)
							min_x =x;
						if(max_y<y)
							max_y=y;
						else if(y!=1 && min_y>y)
							min_y=y;
					}
					
					//cout<<"x:"<<max_x<<"y:"<<max_y;
					x=min(max_x,far[0]);
					y=min(max_y,far[1]);
					//cout<<"x:"<<x<<"y:"<<y;
				//boundRect = Rect(0,0,x,y);*/
		} 
	}
	crop = crop(boundRect);
	//imwrite("crop.jpg", crop);
	return crop;
}


void computeAlignment(Mat image1, Mat image2,int first)
{
/*Warps second image and stiches 2 images and saves into output array for further processing*/
	
	Mat H;
	
	/*Finding Homography matrix to warp the images using RANSAC algorithm*/
	if(first)
	{
		H = findHomography(train, query, CV_RANSAC);
	}
	else
	{
		H = findHomography(query,train, CV_RANSAC);
	}
	int col = 2*min(image1.cols,image2.cols);
	int rows = 2*min(image1.rows,image2.rows);
	
	/*vector<Point2f> obj_corners(4);
	vector<Point2f> scene_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image2.cols, 0 );
    obj_corners[2] = cvPoint( image2.cols, image2.rows ); obj_corners[3] = cvPoint( 0, image2.rows );
	perspectiveTransform( obj_corners, scene_corners, H);*/
	
	//Warping image2 
	warpPerspective(image2,output[outputindex],H,Size(image1.cols+image2.cols,rows));
	Mat half(output[outputindex],Rect(0,0,image1.cols,image1.rows));
	image1.copyTo(half);
	
	//Cropping final image
	output[outputindex] = cropImage(output[outputindex]);
	
	//Rect roi(0, 0, scene_corners[1].x,rows);
    //Mat image_roi = output[outputindex](roi);
	//output[outputindex] = image_roi;
	
	//Dynamically creating final name
	string kpfname = "output";
	stringstream ss;
	ss<<outputindex;
	kpfname = kpfname+ss.str()+".jpg";
	ss.clear();
	imwrite(kpfname,output[outputindex]);
	outputindex++;
}
int main(int argc, char *argv[])
{

	string image_name;
	//Reading nuumber of images
	numimages = atoi(argv[1]);
	
	int imgindex=0;
	int k=2;
	while(imgindex<numimages)
	{
		//Reading input images
		image[imgindex] = imread(argv[k],CV_LOAD_IMAGE_COLOR);
		if(!image[imgindex].data)                              // Check for invalid input
		{
		cout<<"Error in loading the image\n"<< std::endl;
        return -1;
		}
		//Calling detectfeatures to identify keypoints and extract descriptors
		detectFeatures(image[imgindex], imgindex);
		k++;
		imgindex++;
	}
	while(numimages>1)
	{
		//To loop untill final image
		outputindex=0;
		for(imgindex=0;imgindex<numimages;imgindex++)
		{
			for(k=imgindex+1;k<numimages;k++)
			{
			int a=imgindex,b=k;
			match(image[a], image[b], a, b); //matches images, identifies goodmatches,reorders calls computealignment
			
			}
		}
		for(imgindex=0;imgindex<outputindex;imgindex++)
		{
			//Copying the pairs into image array
			output[imgindex].copyTo(image[imgindex]);
			detectFeatures(image[imgindex], imgindex);
		}
		numimages=outputindex;
		//cout<<"\n next round using number of images:"<<numimages;
	}
	cout<<"The output of program is saved in panorama.jpg";
	imwrite("panorama.jpg", output[0]);
}