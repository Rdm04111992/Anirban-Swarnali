import cv2
import numpy as np
import pytesseract
from datetime import datetime

# Load pre-trained YOLO model for vehicle detection
def load_yolo_model():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect vehicles in the frame
def detect_vehicles(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID 2 corresponds to 'car'
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4), boxes

# Extract number plate using OCR
def extract_number_plate(frame, box):
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 8')
    return text.strip()

# Main function to process video feed
def process_traffic_signal(video_path):
    net, classes, output_layers = load_yolo_model()
    cap = cv2.VideoCapture(video_path)
    signal_status = "RED"  # Simulate traffic signal status
    violations = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        indexes, boxes = detect_vehicles(frame, net, output_layers)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            if signal_status == "RED":  # Check if signal is red
                number_plate = extract_number_plate(frame, (x, y, w, h))
                if number_plate:
                    violations.append({
                        "number_plate": number_plate,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "VIOLATION", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Traffic Signal Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save violations to a file
    with open("violations.txt", "w") as f:
        for violation in violations:
            f.write(f"{violation['number_plate']}, {violation['timestamp']}\n")

# Run the program
if __name__ == "__main__":
    process_traffic_signal("traffic_video.mp4")