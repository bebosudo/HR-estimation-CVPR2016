// opencv header files
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

int main(int argc, char** argv) {
    // create a variable to store the image
    cv::Mat image;

    // open the image and store it in the 'image' variable
    // Replace the path with where you have downloaded the image
    image = cv::imread("src/lena.jpg");

    // create a window to display the image
    cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE);

    // display the image in the window created
    cv::imshow("Display window", image);

    // wait for a keystroke
    cv::waitKey(0);
    return 0;
}
