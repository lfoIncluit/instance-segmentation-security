from itertools import count
import cv2
import numpy as np
import imutils
import json
import time
from openvino.inference_engine import IECore

TEST_PATH = "Images"
VIDEO_PATH = "Video/BlindspotFront.mp4"
PAINT = True
CONF = 0.4

redColor = (0, 0, 255)
greenColor = (0, 255, 0)
rectThinkness = 1
alpha = 0.8

instance_segmentation_model_xml = "./model/instance-segmentation-security-1040.xml"
instance_segmentation_model_bin = "./model/instance-segmentation-security-1040.bin"

device = "GPU"


def drawText(frame, scale, rectX, rectY, rectColor, text):

    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 6)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 1
    )


def get_label(index):
    global labels
    return labels.get("coco_list")[index]


def instance_segmentationDetection(
    frame,
    instance_segmentation_neural_net,
    instance_segmentation_execution_net,
    instance_segmentation_input_blob,
    fps,
):

    N, C, H, W = instance_segmentation_neural_net.input_info[
        instance_segmentation_input_blob
    ].tensor_desc.dims
    resized_frame = cv2.resize(frame, (W, H))
    frame_height, frame_width, _ = frame.shape

    # reshape to network input shape
    # Change data layout from HWC to CHW
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

    instance_segmentation_results = instance_segmentation_execution_net.infer(
        inputs={instance_segmentation_input_blob: input_image}
    )

    labels = instance_segmentation_results.get("labels")
    boxes = instance_segmentation_results.get("boxes")
    # masks = instance_segmentation_results.get("masks")

    # print("LABELS: ", labels)
    for i in range(len(labels)):
        if int(labels[i]):
            conf = boxes[i][4]
            if conf < CONF:
                continue

        classId = int(labels[i])
        top_left_x = int(boxes[i][0])
        top_left_y = int(boxes[i][1])
        botton_right_x = int(boxes[i][2])
        botton_right_y = int(boxes[i][3])

        cv2.rectangle(
            resized_frame,
            (top_left_x, top_left_y),
            (botton_right_x, botton_right_y),
            redColor,
            rectThinkness,
        )
        rectW = botton_right_x - top_left_x
        label = get_label(classId)
        drawText(
            resized_frame,
            rectW * 0.02,
            botton_right_x,
            botton_right_y,
            redColor,
            label,
        )
    showImg = cv2.resize(resized_frame, (800, 600))
    drawText(
        showImg,
        1,
        0,
        0,
        greenColor,
        f"FPS : {str(fps)}",
    )
    cv2.imshow("showImg", showImg)


def main():

    ie = IECore()
    labels_file_name = "objects_labels.json"
    labels_file = None
    fps = 0
    frame_count = 0
    global labels
    try:
        with open(labels_file_name, "r") as labels_file:
            labels = json.loads(labels_file.read())

    finally:
        if labels_file:
            labels_file.close()

    instance_segmentation_neural_net = ie.read_network(
        model=instance_segmentation_model_xml, weights=instance_segmentation_model_bin
    )
    if instance_segmentation_neural_net is not None:
        instance_segmentation_execution_net = ie.load_network(
            network=instance_segmentation_neural_net, device_name=device.upper()
        )
        instance_segmentation_input_blob = next(
            iter(instance_segmentation_execution_net.input_info)
        )
        instance_segmentation_neural_net.batch_size = 1

        vidcap = cv2.VideoCapture(VIDEO_PATH)
        success, img = vidcap.read()
        timestamp = time.time()
        while success:
            instance_segmentationDetection(
                img,
                instance_segmentation_neural_net,
                instance_segmentation_execution_net,
                instance_segmentation_input_blob,
                fps,
            )
            new_timestamp = time.time()
            if new_timestamp - timestamp >= 1:
                fps = frame_count
                frame_count = 0
                timestamp = new_timestamp
            else:
                frame_count += 1
            if cv2.waitKey(10) == 27:  # exit if Escape is hit
                break
            success, img = vidcap.read()


if __name__ == "__main__":
    main()
