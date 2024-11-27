import cv2
from ultralytics import YOLO

def count_pepper(video_path, output_path):
    try:
        model = YOLO('./pepper-detection.pt') # load a custom model
    except FileNotFoundError:
        print("Error: Model not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, stream=False)

        for result in results:
            boxes = result.boxes.xyxy
            class_ids = result.boxes.cls
            confidences = result.boxes.conf
            track_ids = result.boxes.id if result.boxes.id is not None else [0]

            for box, cls, conf, id in zip(boxes, class_ids, confidences, track_ids):
                if conf >= 0.7:
                    x1, y1, x2, y2 = map(int, box)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f"{model.names[int(cls)]} {id} {conf:.2f}"

                    cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with annotations to the output video
        out.write(frame)

        # Display the frame (optional)
        # cv2.imshow('Object Tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release resources
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    count_pepper('./input/0001.mp4', './output/0001.mp4')
    count_pepper('./input/0002.mp4', './output/0002.mp4')
    count_pepper('./input/0003.mp4', './output/0003.mp4')
    count_pepper('./input/0004.mp4', './output/0004.mp4')
