#include "CameraCapture.h"
#include <iostream>

CameraCapture::CameraCapture(int cameraIndex)
    : cameraIndex_(cameraIndex), running_(false)
{
}

CameraCapture::~CameraCapture()
{
    stop();
}

bool CameraCapture::initialize()
{
    capture_.open(cameraIndex_);

    if (!capture_.isOpened())
    {
        std::cerr << "Erro ao abrir câmera " << cameraIndex_ << std::endl;
        return false;
    }

    // Configurar resolução (ajuste conforme necessário)
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    capture_.set(cv::CAP_PROP_FPS, 30);

    std::cout << "Câmera inicializada: "
              << capture_.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << capture_.get(cv::CAP_PROP_FRAME_HEIGHT)
              << " @ " << capture_.get(cv::CAP_PROP_FPS) << " FPS" << std::endl;

    return true;
}

void CameraCapture::start()
{
    if (running_)
        return;

    running_ = true;
    captureThread_ = std::thread(&CameraCapture::captureLoop, this);
}

void CameraCapture::stop()
{
    if (!running_)
        return;

    running_ = false;
    if (captureThread_.joinable())
    {
        captureThread_.join();
    }

    if (capture_.isOpened())
    {
        capture_.release();
    }
}

cv::Mat CameraCapture::getLatestFrame()
{
    std::lock_guard<std::mutex> lock(frameMutex_);
    return latestFrame_.clone();
}

void CameraCapture::captureLoop()
{
    cv::Mat frame;

    while (running_)
    {
        if (!capture_.read(frame))
        {
            std::cerr << "Erro ao capturar frame" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frameMutex_);
            latestFrame_ = frame.clone();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
