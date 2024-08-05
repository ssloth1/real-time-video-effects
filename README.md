# real-time-video-effects
This repository contains a real-time video processing tool using OpenCV, featuring effects like scanlines, color noise, chromatic aberration, frame tear, motion blur, tape noise, vignette, and sepia. Includes a user interface with adjustable intensity controls and PyVirtualCam integration for virtual camera output.

## Features

The effects included in the `VideoProcessor` are:

1. High-pass filter
2. Alpha and beta adjustments
3. Scanline effect
4. Color noise
5. Black and white noise
6. Chromatic aberration
7. Frame tear effect
8. Motion blur
9. Tape noise
10. Vignette effect
11. Sepia effect

## Usage
1. The VideoProcessor class is initialized with a video source (default is 0 for the default camera) and a flag indicating whether to use a virtual camera.
2. Trackbars - The program provides a set of trackbars for real-time adjustment of the effects. These trackbars control various parameters of the effects.
3. Virtual Camera - If use_virtual_cam is set to True, the processed frames are sent to a virtual camera using PyVirtualCam. If the virtual camera is not available, it falls back to window display.
4. Save an Image - Press 's' to save the current processed frame as processed_frame.jpg.
5. Quit - Press 'q' to quit the application

## Requirements

- Python 3.12
- OpenCV (`cv2`)
- NumPy (`numpy`)
- PyVirtualCam (optional, for virtual camera functionality)

To install the required libraries, run:

```bash
pip install opencv-python-headless numpy pyvirtualcam
