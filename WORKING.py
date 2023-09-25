import depthai as dai
import cv2
import numpy as np

# Add model path
MODEL_PATH = "INSERT_MODEL_PATH"

# Initialize DepthAI pipeline and device
pipeline = dai.Pipeline()

# Set up RGB camera and its output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setFps(30.0)

# Create a manipulator stage in input image pipeline to resize the image
manip = pipeline.createImageManip()
manip.initialConfig.setResize(448, 224)
cam_rgb.video.link(manip.inputImage)

# Neural network configuration
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(MODEL_PATH)
manip.out.link(nn.input)

# Set up XLinkOut for NN to host communication
xout = pipeline.createXLinkOut()
xout.setStreamName("nn")
nn.out.link(xout.input)

# Set up XLinkOut for RGB video to host communication for visualization
video_queue = pipeline.createXLinkOut()
video_queue.setStreamName("video")
#cam_rgb.video.link(video_queue.input)
manip.out.link(video_queue.input)

# #Decode YOLOv5 output
def decode_yolov5_output(output, conf_thres=0.01):
    # Placeholder for detected objects
    detected_objects = []

    # Grid size
    grid_size = output.shape[-1]

    # Define anchor boxes (update them if you're using custom anchors)
    anchors = [
        [10,13, 16,30, 33,23],
        [30,61, 62,45, 59,119],
        [116,90, 156,198, 373,326]
    ]

    # Decode the raw output
    for i in range(3):  # Iterate over the three scales
        anchor_set = anchors[i]
        for j, anchor in enumerate(anchor_set):
            # Extract bounding box attributes
            x, y, w, h, obj_score = output[0, j, :4, :, :]
            
            # TO-DO: Decode class scores, apply softmax and find the class with max score
            
            # Apply objectness score threshold
            mask = obj_score > conf_thres
            if np.any(mask):
                x, y, w, h = x[mask], y[mask], w[mask], h[mask]
                
                # Convert grid box format to pixel values
                box_pixel = np.array([x * grid_size, y * grid_size, w * grid_size, h * grid_size]).T
                detected_objects.extend(box_pixel)

    return detected_objects

with dai.Device(pipeline) as device:
    nn_queue = device.getOutputQueue(name="nn", maxSize=8, blocking=False)
    video_queue = device.getOutputQueue(name="video", maxSize=8, blocking=False)
    
    while True:
        # Get the raw neural network output
        #in_nn = nn_queue.get().getFirstLayerFp16()

        #in_nn = nn_queue.get().getFirstLayerFp16()
        #in_nn = np.array(in_nn, dtype=np.float32).reshape(448, 224)  # Adjust the shape as required

        # Decode YOLOv5 output
        #detections = decode_yolov5_output(in_nn)

        # Get video frame
        frame = video_queue.get().getCvFrame()

        # # Draw bounding boxes on the frame
        # for detection in detections:
        #     x, y, w, h = map(int, detection)
        #     cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        
        frame = video_queue.get().getCvFrame()
        cv2.imshow("OAK-D Camera Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
