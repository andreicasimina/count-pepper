import cv2
from ultralytics import YOLO

def detect_pepper(input_path, output_path):
    try:
        model = YOLO('./pepper-detection.pt') # load a custom model
    except FileNotFoundError:
        print("Error: Model not found.")
        return

    cap = cv2.imread(input_path)

    frame = cap

    results = model.predict(frame)

    for result in results:
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls
        confidences = result.boxes.conf

        for box, cls, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{model.names[int(cls)]} {conf:.2f}"

            cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with annotations to the output video
        cv2.imwrite(output_path, frame)

        # Display the frame (optional)
        # cv2.imshow('Object Tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release resources
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # count_pepper('./input/0001.mp4', './output/0001.mp4')
    # count_pepper('./input/0002.mp4', './output/0002.mp4')
    # count_pepper('./input/0003.mp4', './output/0003.mp4')
    detect_pepper('./test2.png', './test2-output.png')
