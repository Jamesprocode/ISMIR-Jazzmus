from ultralytics import YOLO

# Load a model
model = YOLO("yolov11s_20241108.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
# TODO
results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # TODO
    result.save(filename="result.jpg")  # save to disk
