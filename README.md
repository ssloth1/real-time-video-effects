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

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- PyVirtualCam (optional, for virtual camera functionality)

To install the required libraries, run:

```bash
pip install opencv-python-headless numpy pyvirtualcam
