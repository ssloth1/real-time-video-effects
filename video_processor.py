import cv2
import numpy as np
import pyvirtualcam # note: pyvirtualcam is not a standard library, you may need to install it with pip
                    # however, if you do not want or need to use a virtual camera, you can remove the dependency
                    # and the code will still work with the window display

"""
The VideoProcessor class processes video frames with a variety of effects.
The effects include:
- High-pass filter
- Alpha and beta adjustments
- Scanline effect
- Color noise
- Black and white noise
- Chromatic aberration
- Frame tear effect
- Motion blur
- Tape noise
- Vignette effect
- Sepia effect
"""
class VideoProcessor:
    def __init__(self, source=0, use_virtual_cam=False):
        self.cap = cv2.VideoCapture(source)
        self.kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
        self.ksize_index = 0
        self.alpha = 1
        self.beta = 0
        self.color_channel = 127
        self.noise_intensity = 0
        self.scanline_intensity = 0
        self.chromatic_aberration_intensity = 0
        self.tape_noise_intensity = 0
        self.motion_blur_strength = 0
        self.frame_tear_intensity = 0
        self.color_mode = 0
        self.bw_noise_intensity = 0
        self.vignette_intensity = 0
        self.sepia_intensity = 0
        self.use_virtual_cam = use_virtual_cam

    # Process a video frame with a variety of effects
    def process_frame(self, frame):
        frame = self.apply_color_mode(frame)
        frame = self.apply_alpha_beta_adjustments(frame)
        frame = self.apply_scanline_effect(frame)
        frame = self.add_color_noise(frame)
        frame = self.add_bw_noise(frame)
        frame = self.add_chromatic_aberration(frame)
        frame = self.add_frame_tear_effect(frame)
        frame = self.apply_motion_blur(frame)
        frame = self.add_tape_noise(frame)
        frame = self.apply_vignette(frame)
        frame = self.apply_sepia(frame)
        return frame

    # apply color mode to the frame
    def apply_color_mode(self, frame):
        if self.color_mode == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif self.color_mode == 2 and len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if self.ksize_index > 0:
                frame = self.apply_high_pass_filter(frame)
        return frame

    # apply high-pass filter to the frame,
    # this is necessary for the color channel effect, as it requires a high-pass contrast image
    def apply_high_pass_filter(self, frame):
        ksize = self.kernel_sizes[self.ksize_index]
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        low_pass = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        high_pass = cv2.subtract(frame, low_pass)
        high_pass_contrast = cv2.convertScaleAbs(cv2.cvtColor(high_pass, cv2.COLOR_BGR2GRAY), alpha=self.alpha, beta=self.beta)
        colored_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        if self.color_channel < 128: # Apply color channel to high-pass contrast
            ratio = self.color_channel / 127.0
            colored_frame[:, :, 0] = (1 - ratio) * high_pass_contrast
            colored_frame[:, :, 1] = ratio * high_pass_contrast
        else: # Apply color channel to original frame
            ratio = (self.color_channel - 128) / 127.0
            colored_frame[:, :, 1] = (1 - ratio) * high_pass_contrast
            colored_frame[:, :, 2] = ratio * high_pass_contrast
        return colored_frame

    # apply alpha and beta adjustments to the frame
    def apply_alpha_beta_adjustments(self, frame):
        return cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)

    # apply scanline effect to the frame
    def apply_scanline_effect(self, frame):
        if self.scanline_intensity == 0:
            return frame
        rows, cols, _ = frame.shape
        scanline_frame = frame.copy()  
        for i in range(0, rows, 2): # for every other row, darken the row, creating a scanline effect
            scanline_frame[i:i + 1, :] = (scanline_frame[i:i + 1, :] * (1 - self.scanline_intensity)).astype(np.uint8) # darken the row
        return scanline_frame

    # add color noise to the frame
    def add_color_noise(self, frame):
        if self.noise_intensity == 0:
            return frame
        noise = np.random.randint(0, 256, frame.shape, dtype=np.uint8) # generate random values for noise
        return cv2.addWeighted(frame, 1 - self.noise_intensity, noise, self.noise_intensity, 0) # add noise to the frame

    # add black and white noise to the frame
    def add_bw_noise(self, frame):
        if self.bw_noise_intensity == 0:
            return frame
        noise = np.random.randint(0, 256, frame.shape[:2], dtype=np.uint8) # generate randome values for the black and white noise
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR) # convert the noise to a 3-channel image
        return cv2.addWeighted(frame, 1 - self.bw_noise_intensity, noise, self.bw_noise_intensity, 0) # add noise to the frame

    # add chromatic aberration to the frame
    def add_chromatic_aberration(self, frame):
        if self.chromatic_aberration_intensity == 0:
            return frame
        shift = int(self.chromatic_aberration_intensity * 5) # shift the blue and red channels by a random amount
        b, g, r = cv2.split(frame) # split the frame into its blue, green, and red channels
        b_shifted = np.roll(b, shift, axis=1) # shift the blue
        r_shifted = np.roll(r, -shift, axis=1) # shift the red
        return cv2.merge((b_shifted, g, r_shifted)) # merge the shifted channels back together

    # add frame tear effect to the frame
    def add_frame_tear_effect(self, frame):
        if self.frame_tear_intensity == 0:
            return frame
        rows, cols, _ = frame.shape
        tear_frame = frame.copy()
        scaled_intensity = int(self.frame_tear_intensity * 10) 
        for _ in range(scaled_intensity): # create a random number of tears in the frame
            tear_y = np.random.randint(rows) # randomly select a row to tear
            tear_height = np.random.randint(1, rows // 10) # randomly select the height of the tear
            tear_frame[tear_y:tear_y + tear_height, :] = np.roll(tear_frame[tear_y:tear_y + tear_height, :], np.random.randint(-10, 10), axis=1) # roll the torn section of the frame
        return tear_frame

    # apply motion blur to the frame
    def apply_motion_blur(self, frame):
        if self.motion_blur_strength == 0:
            return frame
        kernel = np.zeros((self.motion_blur_strength, self.motion_blur_strength)) 
        kernel[int((self.motion_blur_strength - 1) / 2), :] = np.ones(self.motion_blur_strength) # create a motion blur kernel
        kernel /= self.motion_blur_strength # normalize the kernel
        return cv2.filter2D(frame, -1, kernel) # apply the motion blur kernel to the frame
    
    # add tape noise to the frame
    def add_tape_noise(self, frame):
        if self.tape_noise_intensity == 0:
            return frame
        rows, cols, _ = frame.shape
        tape_noise_frame = frame.copy()
        num_lines = int(self.tape_noise_intensity * 10)
        for _ in range(num_lines): # create a random number of tape noise lines
            y = np.random.randint(0, rows) # randomly select a row to add tape noise
            line_thickness = np.random.randint(1, 3) # randomly select the thickness of the line
            alpha = np.random.uniform(0.3, 0.7) # randomly select the alpha value
            overlay = tape_noise_frame.copy() # create an overlay to add the tape noise to
            cv2.line(overlay, (0, y), (cols, y), (0, 0, 0), line_thickness) # draw the tape noise line
            tape_noise_frame = cv2.addWeighted(overlay, alpha, tape_noise_frame, 1 - alpha, 0) # add the tape noise line to the frame
        return tape_noise_frame

    # applies a vignetter effect to the frame, that darkens the edges of the frame
    def apply_vignette(self, frame):
        if self.vignette_intensity == 0:
            return frame
        rows, cols = frame.shape[:2]
        xk = cv2.getGaussianKernel(cols, cols / 2) # create a Gaussian kernel for the x-axis
        yk = cv2.getGaussianKernel(rows, rows / 2) # create a Gaussian kernel for the y-axis
        resulting_kernel = yk * xk.T # create a 2D Gaussian kernel
        mask = 255 * resulting_kernel / np.linalg.norm(resulting_kernel) # normalize the kernel
        vignette = np.copy(frame) # create a copy of the frame
        for i in range(3): # apply the vignette to each channel of the frame
            vignette[:,:,i] = vignette[:,:,i] * mask
        return cv2.addWeighted(frame, 1 - self.vignette_intensity, vignette, self.vignette_intensity, 0)

    # applies a sepia effect to the frame for a vintage look
    def apply_sepia(self, frame):
        if self.sepia_intensity == 0:
            return frame
        # I learned how to do this from https://stackoverflow.com/questions/23802725/using-numpy-to-apply-a-sepia-effect-to-a-3d-array
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, sepia_filter)
        return cv2.addWeighted(frame, 1 - self.sepia_intensity, sepia_frame, self.sepia_intensity, 0)

def on_trackbar_change(value):
    global processor
    try:
        processor.ksize_index = cv2.getTrackbarPos('Kernel Size', 'Trackbars')
        processor.alpha = cv2.getTrackbarPos('Alpha', 'Trackbars') / 10.0
        processor.beta = cv2.getTrackbarPos('Beta', 'Trackbars')
        processor.color_channel = cv2.getTrackbarPos('Color Channel', 'Trackbars')
        processor.noise_intensity = cv2.getTrackbarPos('Noise Intensity', 'Trackbars') / 100.0
        processor.scanline_intensity = cv2.getTrackbarPos('Scanline Intensity', 'Trackbars') / 100.0
        processor.chromatic_aberration_intensity = cv2.getTrackbarPos('Chromatic Aberration Intensity', 'Trackbars') / 100.0
        processor.tape_noise_intensity = cv2.getTrackbarPos('Tape Noise Intensity', 'Trackbars') / 10.0
        processor.motion_blur_strength = cv2.getTrackbarPos('Motion Blur Strength', 'Trackbars')
        processor.frame_tear_intensity = cv2.getTrackbarPos('Frame Tear Intensity', 'Trackbars') / 10.0
        processor.color_mode = cv2.getTrackbarPos('Color Mode', 'Trackbars')
        processor.bw_noise_intensity = cv2.getTrackbarPos('Black and White Noise Intensity', 'Trackbars') / 100.0
        processor.vignette_intensity = cv2.getTrackbarPos('Vignette Intensity', 'Trackbars') / 100.0
        processor.sepia_intensity = cv2.getTrackbarPos('Sepia Intensity', 'Trackbars') / 100.0
    except cv2.error as e:
        print(f"Trackbar error: {e}")

def main():
    global processor
    use_virtual_cam = False
    processor = VideoProcessor(0, use_virtual_cam)
    
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Video Processor')
    
    cv2.createTrackbar('Color Mode', 'Trackbars', 0, 2, on_trackbar_change)
    cv2.createTrackbar('Color Channel', 'Trackbars', 127, 255, on_trackbar_change)
    cv2.createTrackbar('Kernel Size', 'Trackbars', 4, len(processor.kernel_sizes) - 1, on_trackbar_change)
    cv2.createTrackbar('Alpha', 'Trackbars', 10, 30, on_trackbar_change)
    cv2.createTrackbar('Beta', 'Trackbars', 0, 255, on_trackbar_change)
    cv2.createTrackbar('Noise Intensity', 'Trackbars', 0, 100, on_trackbar_change)
    cv2.createTrackbar('Black and White Noise Intensity', 'Trackbars', 0, 100, on_trackbar_change)
    cv2.createTrackbar('Tape Noise Intensity', 'Trackbars', 0, 10, on_trackbar_change)
    cv2.createTrackbar('Scanline Intensity', 'Trackbars', 0, 100, on_trackbar_change)
    cv2.createTrackbar('Chromatic Aberration Intensity', 'Trackbars', 0, 100, on_trackbar_change)
    cv2.createTrackbar('Motion Blur Strength', 'Trackbars', 0, 10, on_trackbar_change)
    cv2.createTrackbar('Frame Tear Intensity', 'Trackbars', 0, 10, on_trackbar_change)
    cv2.createTrackbar('Vignette Intensity', 'Trackbars', 0, 100, on_trackbar_change)
    cv2.createTrackbar('Sepia Intensity', 'Trackbars', 0, 100, on_trackbar_change)
    
    cv2.setTrackbarPos('Kernel Size', 'Trackbars', processor.ksize_index)
    cv2.setTrackbarPos('Alpha', 'Trackbars', int(processor.alpha * 10))
    cv2.setTrackbarPos('Beta', 'Trackbars', processor.beta)
    cv2.setTrackbarPos('Color Channel', 'Trackbars', processor.color_channel)
    cv2.setTrackbarPos('Noise Intensity', 'Trackbars', int(processor.noise_intensity * 100))
    cv2.setTrackbarPos('Scanline Intensity', 'Trackbars', int(processor.scanline_intensity * 100))
    cv2.setTrackbarPos('Chromatic Aberration Intensity', 'Trackbars', int(processor.chromatic_aberration_intensity * 100))
    cv2.setTrackbarPos('Tape Noise Intensity', 'Trackbars', int(processor.tape_noise_intensity * 10))
    cv2.setTrackbarPos('Motion Blur Strength', 'Trackbars', processor.motion_blur_strength)
    cv2.setTrackbarPos('Frame Tear Intensity', 'Trackbars', int(processor.frame_tear_intensity * 10))
    cv2.setTrackbarPos('Color Mode', 'Trackbars', processor.color_mode)
    cv2.setTrackbarPos('Black and White Noise Intensity', 'Trackbars', int(processor.bw_noise_intensity * 100))
    cv2.setTrackbarPos('Vignette Intensity', 'Trackbars', int(processor.vignette_intensity * 100))
    cv2.setTrackbarPos('Sepia Intensity', 'Trackbars', int(processor.sepia_intensity * 100))
    
    # Check if virtual camera is available, otherwise fall back to window display
    cam = None
    if processor.use_virtual_cam:
        try:
            cam = pyvirtualcam.Camera(width=640, height=480, fps=30, backend='obs')
            print(f'Using virtual camera: {cam.device}')
        except pyvirtualcam.CameraError:
            print('Virtual camera not available. Falling back to window display.')

    # Process video frames
    while processor.cap.isOpened():
        ret, frame = processor.cap.read()
        if not ret:
            break
        
        processed_frame = processor.process_frame(frame)
        
        # Display processed frame
        if cam:
            cam.send(processed_frame)
            cam.sleep_until_next_frame()
        
        cv2.imshow('Video Processor', processed_frame)
        key = cv2.waitKey(1)

        # Handle key presses, 'q' to quit, 's' to save frame
        if key == ord('s'):
            save_path = "processed_frame.jpg"
            cv2.imwrite(save_path, processed_frame)
            print(f"Frame saved to {save_path}")
        elif key != -1:
            break
    
    if cam:
        cam.close()
    processor.cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
