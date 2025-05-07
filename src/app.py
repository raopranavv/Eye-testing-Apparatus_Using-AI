import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if not cap.isOpened():
        print("Error")
        return None

    print("'space' to capture, to exit 'escape'")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (fx, fy, fw, fh) in faces:
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (0, 255, 0), 2)

        cv2.imshow('Capture Image', frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("Closing without capture.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key % 256 == 32:
            img_name = "captured_image.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} captured.")
            cap.release()
            cv2.destroyAllWindows()
            return img_name

def is_eye_cloudy(eye_region, eye_count):
    edges = cv2.Canny(eye_region, 100, 200)
    edge_image_path = f"edge_detected_eye_{eye_count}.png"
    cv2.imwrite(edge_image_path, edges)

    edge_area = np.sum(edges != 0)
    cloudiness_ratio = edge_area / (eye_region.shape[0] * eye_region.shape[1])
    cloudiness_threshold = 0.2
    is_cloudy = cloudiness_ratio > cloudiness_threshold
    return is_cloudy, cloudiness_ratio, edge_area, edge_image_path

def detect_cataract(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    eye_details = []
    eye_count = 0

    for (fx, fy, fw, fh) in faces:
        face_roi = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

        for (ex, ey, ew, eh) in eyes:
            eye_count += 1
            eye_region = face_roi[ey:ey+eh, ex:ex+ew]
            eye_image_path = f"detected_eye_{eye_count}.png"
            cv2.imwrite(eye_image_path, eye_region)

            is_cloudy, cloudiness_ratio, edge_area, edge_image_path = is_eye_cloudy(eye_region, eye_count)
            eye_info = {
                "position": (fx+ex, fy+ey),
                "size": (ew, eh),
                "cloudiness_ratio": cloudiness_ratio,
                "edge_area": edge_area,
                "cataract_detected": is_cloudy,
                "eye_image": eye_image_path,
                "edge_image": edge_image_path
            }
            eye_details.append(eye_info)

    return eye_details

image_path = capture()
if image_path:
    eye_data = detect_cataract(image_path)
    edge_areas = [eye['edge_area'] for eye in eye_data]

    plt.bar(range(len(edge_areas)), edge_areas)
    plt.xlabel('Eye Number')
    plt.ylabel('Edge Area')
    plt.title('Edge Areas in Detected Eyes')
    plt.show()

    for eye in eye_data:
        print(f"Eye at position {eye['position']} with size {eye['size']}:")
        if eye['cataract_detected']:
            print(f"  - Cataract likely detected (Cloudiness Ratio: {eye['cloudiness_ratio']}, Edge Area: {eye['edge_area']})")
            print(f"  - Eye Image: {eye['eye_image']}")
            print(f"  - Edge Detected Image: {eye['edge_image']}")
        else:
            print(f"  - No cataract detected (Cloudiness Ratio: {eye['cloudiness_ratio']}, Edge Area: {eye['edge_area']})")
            print(f"  - Eye Image: {eye['eye_image']}")
            print(f"  - Edge Detected Image: {eye['edge_image']}")