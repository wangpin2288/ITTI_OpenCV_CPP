#include "ITTI.cpp"

int main()
{
	string filename = "0_0_147.jpg";
	Mat img = imread(filename);
 
	ITTI itti(img);
	Mat salMap = itti.getSalMap();

	namedWindow("WindowOrg",1);
	imshow("WindowOrg",img);

	namedWindow("WindowNew",1);
	imshow("WindowNew",salMap);
	waitKey(0);

	return 0;
}












