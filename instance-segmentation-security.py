import cv2
import numpy as np
import imutils
from openvino.inference_engine import IECore

TEST_PATH = "Images"
VIDEO_PATH = "Video/BlindspotFront.mp4"
PAINT = True
CONF = 0.4

pColor = (0, 0, 255)
rectThinkness = 1
alpha = 0.8

instance_segmentation_model_xml = "./model/instance-segmentation-security-1040.xml"
instance_segmentation_model_bin = "./model/instance-segmentation-security-1040.bin"

device = "CPU"


def drawText(frame, scale, rectX, rectY, rectColor, text):

    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3
    )


def instance_segmentationDetection(
    frame,
    instance_segmentation_neural_net,
    instance_segmentation_execution_net,
    instance_segmentation_input_blob,
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

    print("LABELS: ", labels)
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
            pColor,
            rectThinkness,
        )
        rectW = botton_right_x - top_left_x
        drawText(
            resized_frame,
            rectW * 0.008,
            botton_right_x,
            botton_right_y,
            pColor,
            str(classId),
        )

    showImg = cv2.resize(resized_frame, (800, 600))
    cv2.imshow("showImg", showImg)


def main():

    ie = IECore()

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

        while success:
            instance_segmentationDetection(
                img,
                instance_segmentation_neural_net,
                instance_segmentation_execution_net,
                instance_segmentation_input_blob,
            )
            if cv2.waitKey(10) == 27:  # exit if Escape is hit
                break
            success, img = vidcap.read()


if __name__ == "__main__":
    main()
