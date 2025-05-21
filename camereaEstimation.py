import cv2
import numpy as np
from hitnet import HitNet, ModelType

# Select model type
model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
    model_path = "models/middlebury_d400/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.flyingthings:
    model_path = "models/flyingthings_finalpass_xl/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.eth3d:
    model_path = "models/eth3d/saved_model_480x640/model_float32.onnx"

# Initialize model
depth_estimator = HitNet(model_path, model_type)

# Open the webcam (assuming stereo camera setup with two video streams)
left_camera_index = 0  # Index for the left camera
right_camera_index = 1  # Index for the right camera

left_cap = cv2.VideoCapture(left_camera_index)
right_cap = cv2.VideoCapture(right_camera_index)

# Set resolution to VGA (640x480)
left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not left_cap.isOpened() or not right_cap.isOpened():
    print("Error: Unable to open one or both cameras.")
    exit()

cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)

while True:
    # Capture frames from both cameras
    ret_left, left_img = left_cap.read()
    ret_right, right_img = right_cap.read()

    if not ret_left or not ret_right:
        print("Error: Unable to read frames from one or both cameras.")
        break

    # Estimate the depth
    disparity_map = depth_estimator(left_img, right_img)

    # Generate a color disparity map
    color_disparity = depth_estimator.draw_disparity()

    # Combine the left image and the disparity map for visualization
    combined_image = np.hstack((left_img, color_disparity))

    # Display the result
    cv2.imshow("Estimated disparity", combined_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()
