import cv2
import numpy as np
import time  # 追加
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

# Load video files
left_video_path = "./DepthMapL.mp4"  # Path to the left video
right_video_path = "./DepthMapR.mp4"  # Path to the right video

left_cap = cv2.VideoCapture(left_video_path)
right_cap = cv2.VideoCapture(right_video_path)

# Check if videos are opened successfully
if not left_cap.isOpened() or not right_cap.isOpened():
    print("Error: Unable to open one or both video files.")
    exit()

# Get video properties
frame_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(left_cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter for output
output_video_path = "output_disparity.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))

while True:
    # Read frames from both videos
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()

    if not ret_left or not ret_right:
        print("End of video or error reading frames.")
        break

    # 計測開始
    start_time = time.time()

    # Estimate the depth
    disparity_map = depth_estimator(left_frame, right_frame)

    # Generate a color disparity map
    color_disparity = depth_estimator.draw_disparity()

    # 計測終了
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time for this frame: {processing_time:.4f} seconds")

    # Combine the left frame and the disparity map for visualization
    combined_frame = np.hstack((left_frame, color_disparity))

    # Write the combined frame to the output video
    out.write(combined_frame)

    # Display the result (optional)
    cv2.imshow("Estimated disparity", combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
left_cap.release()
right_cap.release()
out.release()
cv2.destroyAllWindows()
