#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <set>

using namespace cv;
using namespace std;

struct Pos {
    int x;
    int y;
    bool operator<(const Pos& other) const {
        if (x != other.x)
            return x < other.x;
        return y < other.y;
    }
};

Pos getNextUnvisited(const cv::Mat& visited, Pos prevPos);
Pos step(const cv::Mat& img, const cv::Mat& visited, Pos pos);
cv::Vec3b valueToColour(int value, int maxValue, cv::Vec3b colour1, cv::Vec3b colour2);
int videoFrame(int rows, int cols, const cv::Mat& visited, int colourCount, Pos pos, Pos prevPos, int counter, cv::Vec3b colour1, cv::Vec3b colour2);

static vector<Pos> adjacents = { Pos{-1,-1},  Pos{0,-1},  Pos{1,-1},  Pos{-1,0},  Pos{1,0},  Pos{-1,1},  Pos{0,1},  Pos{1,1} };

bool makeVideo = false;
int fps = 300;
VideoWriter  writer(
    "output.avi",
    VideoWriter::fourcc('M', 'J', 'P', 'G'),
    fps,
    Size(500, 500)
);

/*Noise level variables*/

//Set best difference to transistion bound value (if no neighbour is within bounds then jump)
// Higher = longer paths so more squiggly
int pathCutoff = 200;
//num of pixels in image
int rows = 300; int cols = 300;
//value between 2 and 256, more = noisier 
int noiseMapRange = 50;
//Colour gradient end points
cv::Vec3b colour1(230, 64, 222);
cv::Vec3b colour2(240, 230, 66);
//Shuffle breadth first search priority list 
bool shuffleSearcher = false;




int main()
{

    srand((unsigned)time(0));

    Mat noiseMap(rows, cols, CV_8U);
    for (int y = 0; y < noiseMap.rows; y++)
    {
        for (int x = 0; x < noiseMap.cols; x++)
        {
            noiseMap.at<uchar>(y, x) = rand() % noiseMapRange;
        }
    }


    Pos pos{ rand() % rows, rand() % cols };
    Pos prevPos = pos;
    int colourCount = 1;
    // 0 = unvisited, >0 = colour group
    Mat visited(rows, cols, CV_32S, Scalar(0));
    Mat img(rows, cols, CV_8UC3, Scalar(0, 0, 0));

    int counter = 0;
    while (true) {
        counter++;
        visited.at<int>(pos.y, pos.x) = colourCount;


        if (makeVideo && counter%2 == 0) {
            videoFrame(rows, cols, visited, colourCount, pos, prevPos, counter, colour1, colour2) == -1;
        }


        prevPos = pos;
        pos = step(noiseMap, visited, pos);

        if (pos.x == -1) {
            pos = getNextUnvisited(visited, prevPos);
            if (pos.x == -1) {
                break;
            }
            colourCount++;

        }

    }


    for (int y = 0; y < visited.rows; y++)
    {
        for (int x = 0; x < visited.cols; x++)
        {
            img.at<Vec3b>(y, x) = valueToColour(visited.at<int>(y, x), colourCount, colour1, colour2);

        }
    }


    if (makeVideo) {
        videoFrame(rows, cols, visited, colourCount, prevPos, prevPos, counter, colour1, colour2) == -1;
        writer.release(); 
    }
    Mat resized;
    resize(img, resized, Size(500, 500), 0, 0, INTER_NEAREST);
    cv::imwrite("output.png", resized);
    imshow("Random Image", resized);
    waitKey(0);

    return 0;
}

cv::Vec3b valueToColour( int value, int maxValue,cv::Vec3b colour1,cv::Vec3b colour2)
{
    float t = min((float)value / maxValue, 1.0f);

    int b = colour2[0] + (colour1[0] - colour2[0]) * t;
    int g = colour2[1] + (colour1[1] - colour2[1]) * t;
    int r = colour2[2] + (colour1[2] - colour2[2]) * t;

    return cv::Vec3b(b, g, r);
}

// Breadth first search to find nearest unvisited space
Pos getNextUnvisited(const cv::Mat& visited, Pos prevPos) {

    std::queue<Pos> posQueue;
    posQueue.push(prevPos);
    std::set<Pos> checkedSet;
    checkedSet.insert(prevPos);


    if (shuffleSearcher) {
        random_shuffle(adjacents.begin(), adjacents.end());
    }
    while (posQueue.empty() == 0) {
        for (int i = 0; i < 8; i++)
        {
            Pos current = posQueue.front();
            int nx = current.x + adjacents[i].x;
            int ny = current.y + adjacents[i].y;

            if (nx < 0 || ny < 0 || nx >= visited.cols || ny >= visited.rows)
            {
                continue;
            }

            Pos next{ nx, ny };        
            if (checkedSet.find(next) != checkedSet.end())
            {
                continue;
            }

            if (visited.at<int>(ny, nx) == 0) {
                return next;
            }
            checkedSet.insert(next);
            posQueue.push(next);        
        }
        posQueue.pop();
    }

    return Pos{ -1,-1 };
}
Pos step(const cv::Mat& img, const cv::Mat& visited, Pos pos) {
    int currentVal = img.at<uchar>(pos.y, pos.x);

    int bestDifference = 1000;
    Pos bestPos{ -1,-1 };

    if(shuffleSearcher){
        random_shuffle(adjacents.begin(), adjacents.end());
    }
    for (int i = 0; i < 8; i++)
    {
        int nx = pos.x + adjacents[i].x;
        int ny = pos.y + adjacents[i].y;

        if (nx < 0 || ny < 0 || nx >= img.cols || ny >= img.rows)
        {
            continue;
        }

        int neighbourVal = img.at<uchar>(ny, nx);

        int difference = abs(currentVal - neighbourVal);

        if (difference < bestDifference && visited.at<int>(ny, nx) == 0) 
        {
            bestDifference = difference;

            bestPos.x = nx;
            bestPos.y = ny;
        }
    }

    //Set best difference to transistion bound value (if no neighbour is within bounds then jump)

    if (bestDifference > pathCutoff)
        return Pos{ -1, -1 };
    return bestPos;


}
Mat frame(rows, cols, CV_8UC3, Scalar(0,0,0));
Mat resized(500, 500, CV_8UC3);
int videoFrame(int rows, int cols, const cv::Mat& visited, int colourCount, Pos pos, Pos prevPos, int counter, cv::Vec3b colour1, cv::Vec3b colour2) {

    frame.at<Vec3b>(prevPos.y, prevPos.x) = valueToColour(visited.at<int>(prevPos.y, prevPos.x), rows * cols/15, colour1, colour2);
    frame.at<Vec3b>(pos.y, pos.x) = valueToColour(visited.at<int>(pos.y, pos.x), rows * cols / 15, colour1, colour2);
    cv::resize(frame, resized, resized.size(), 0, 0, INTER_NEAREST);

    writer.write(resized);


    return 0;
}