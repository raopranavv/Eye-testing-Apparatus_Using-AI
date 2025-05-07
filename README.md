# Eye-testing-Apparatus_Using-AI

---

# 👁️ Eye Testing Apparatus Using AI

*By \[Your Name(s)], \[Roll Number(s)]*

---

## 🧠 Problem Statement

Early detection of cataracts and other ocular anomalies can greatly improve the chances of successful treatment and vision restoration. However, access to ophthalmic diagnostic equipment remains limited in many rural or underprivileged areas. This project proposes a lightweight, AI-assisted solution that can preliminarily assess eye health using just a webcam and computer vision.

---

## 🎯 Objectives

* To develop an AI-powered system for basic eye health screening.
* To capture and process eye images in real-time using OpenCV.
* To analyze the cloudiness of eyes using edge detection to predict cataract presence.
* To visualize the processed data to aid interpretation.

---

## 🛠️ Methodology / Architecture / Tools Used

### Architecture:

1. **Image Capture**:

   * Utilizes webcam feed via `OpenCV`.
   * Detects faces and eye regions using Haar cascades.

2. **Preprocessing**:

   * Converts the captured image to grayscale.
   * Detects eyes using Haar cascades.

3. **Analysis**:

   * Applies Canny Edge Detection on eye regions.
   * Calculates "cloudiness ratio" (edge area vs eye area) to infer possible cataracts.

4. **Visualization**:

   * Displays a bar graph of edge areas for each eye.
   * Outputs detailed results for each eye with image paths.

### Tools & Libraries:

* Python
* OpenCV
* NumPy
* Matplotlib

---

## 🚀 Key Features

* 🎥 **Real-time webcam-based eye capture**
* 👁️ **Face and eye detection using Haar cascade classifiers**
* 🧪 **AI-assisted cataract detection using edge analysis**
* 📊 **Graphical visualization of edge data**
* 💾 **Saves cropped eye and edge-detected images for reference**
* 📋 **Detailed textual report of each eye with cloudiness metrics**

---

## 📸 Screenshots

| Face & Eye Detection                           | Edge Detection Result                               |
| ---------------------------------------------- | --------------------------------------------------- |
| ![Face detection sample](./captured_image.png) | ![Edge detection sample](./edge_detected_eye_1.png) |

*(Replace above image paths with your actual screenshots once captured)*

---

## ✅ Conclusion & Future Work

This project demonstrates a practical application of AI and computer vision in ophthalmology. Although not a replacement for professional diagnosis, the apparatus can serve as a preliminary screening tool, particularly in remote regions with limited access to healthcare.

### Future Enhancements:

* Integration with mobile devices for remote screening.
* Use of deep learning models (e.g., CNNs) for higher accuracy.
* Deployment as a web or mobile application.
* Adding support for other eye conditions like glaucoma or diabetic retinopathy.

---

## 🧪 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/Eye-testing-Apparatus_Using-AI.git
   cd Eye-testing-Apparatus_Using-AI
   ```

2. Install requirements:

   ```bash
   pip install opencv-python numpy matplotlib
   ```

3. Run the application:

   ```bash
   python eye_test_app.py
   ```

> **Note**: Press `SPACE` to capture an image, `ESC` to exit.

---

Would you like me to create a badge section (e.g., Python version, OpenCV version, etc.) or a `LICENSE` file for this as well?
