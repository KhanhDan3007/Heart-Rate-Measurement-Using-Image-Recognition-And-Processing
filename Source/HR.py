import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Khoi tao danh sach luu tin hieu 
signal_buffer = []
buffer_size = 300  

# Buffer làm mượt BPM
bpm_buffer = []
bpm_buffer_size = 10

plt.figure(figsize=(15, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]

        # ===== ROI trán =====
        side = w // 5
        roi_x = x + w//2 - side//2
        roi_y = y + h//6 - side//2

        # ===== ROI má trái =====
        roi_l_x = x + w//4 - side//2
        roi_l_y = y + (h//2 + 15) - side//2

        # ===== ROI má phải =====
        roi_r_x = x + 3*w//4 - side//2
        roi_r_y = y + (h//2 + 15) - side//2

        # Vẽ bounding box khuôn mặt
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

        # Vẽ 3 ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + side, roi_y + side), (0,255,0), 2)       # trán
        cv2.rectangle(frame, (roi_l_x, roi_l_y), (roi_l_x + side, roi_l_y + side), (255,255,0), 2) # má trái
        cv2.rectangle(frame, (roi_r_x, roi_r_y), (roi_r_x + side, roi_r_y + side), (255,255,0), 2) # má phải

        #************** Lấy ROI **************
        roi_forehead = frame[roi_y: roi_y + side, roi_x: roi_x + side]
        roi_left = frame[roi_l_y: roi_l_y + side, roi_l_x: roi_l_x + side]
        roi_right = frame[roi_r_y: roi_r_y + side, roi_r_x: roi_r_x + side]

        if roi_forehead.size == 0 or roi_left.size == 0 or roi_right.size == 0:
            continue

        #************** Trích xuất tín hiệu **************
        green_f = np.mean(roi_forehead[:,:,1])
        green_l = np.mean(roi_left[:,:,1])
        green_r = np.mean(roi_right[:,:,1]) 

        # ===== Combine 50% trán + 25% mỗi má =====
        green_mean = 0.5 * green_f + 0.25 * green_l + 0.25 * green_r
        
        plt.clf()

        # ======= 1. Raw signal =======
        plt.subplot(1,4,1)
        plt.plot(signal_buffer)
        plt.xlabel("Data")
        plt.ylabel("G value")
        plt.title("Mean")

        signal_buffer.append(green_mean)

        if len(signal_buffer) > buffer_size:
            signal_buffer.pop(0)

        #************** Lọc tín hiệu **************
        if len(signal_buffer) == buffer_size:

            fs = 30.0 

            # ======= 2. Bandpass Filter =======
            b, a = signal.butter(3, [1.0/(fs/2), 3.0/(fs/2)], btype='bandpass')
            filtered = signal.filtfilt(b, a, signal_buffer)

            plt.subplot(1,4,2)
            plt.plot(filtered)
            plt.title("Bandpass Filtered Signal (1-3 Hz)")
            plt.xlabel("Data")
            plt.ylabel("G value")

            # ======= 3. Detrend =======
            filtered = signal.detrend(filtered)

            plt.subplot(1,4,3)
            plt.plot(filtered)
            plt.title("Detrended Signal")
            plt.xlabel("Data")
            plt.ylabel("G value")

            # ======= 4. FFT =======
            freqs = np.fft.rfftfreq(len(filtered), d=1/fs)
            fft_magnitude = np.abs(np.fft.rfft(filtered))

            fft_magnitude = signal.medfilt(fft_magnitude, kernel_size=5)

            mask = (freqs >= 1.0) & (freqs <= 3.0)

            plt.subplot(1,4,4)
            plt.plot(freqs[mask], fft_magnitude[mask])
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.title("FFT Spectrum (Heart Rate)")
            plt.pause(0.001)

            #************** Tìm peak **************
            freq_range = freqs[mask]
            fft_range = fft_magnitude[mask]

            peak_idx = np.argmax(fft_range)
            peak_freq = freq_range[peak_idx]
            heart_rate = peak_freq * 60

            #************** Làm mượt BPM **************
            bpm_buffer.append(heart_rate)
            if len(bpm_buffer) > bpm_buffer_size:
                bpm_buffer.pop(0)

            smooth_bpm = np.mean(bpm_buffer)

            cv2.putText(frame, f"Heart Rate: {smooth_bpm:.1f} BPM", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            print(f"Heart Rate: {smooth_bpm:.2f} BPM")

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        plt.savefig("capture_plot.png")
        cv2.imwrite("capture_face.png", frame)
        print("Saved plot! + Camera frame")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()