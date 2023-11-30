# OBSS SAHI Tool
# Code written by AnNT, 2023.

import random
import logging
from hashlib import sha256
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements


def nms(bounding_boxes, confidences, threshold):
    """
    Args:
        bounding_boxes: np.array([(x1, y1, x2, y2), ...])
        confidences: np.array(conf1, conf2, ...),数量需要与bounding box一致,并且一一对应
        threshold: IOU阀值,若两个bounding box的交并比大于该值，则置信度较小的box将会被抑制

    Returns:
        bounding_boxes: 经过NMS后的bounding boxes
    """
    len_bound = bounding_boxes.shape[0]
    len_conf = confidences.shape[0]
    if len_bound != len_conf:
        raise ValueError("Bounding box 与 Confidence 的数量不一致")
    if len_bound == 0:
        return []
    bounding_boxes, confidences = bounding_boxes.astype(np.float32), np.array(confidences)

    x1, y1, x2, y2 = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)

    pick = []
    while len(idxs) > 0:
        # 因为idxs是从小到大排列的，last_idx相当于idxs最后一个位置的索引
        last_idx = len(idxs) - 1
        # 取出最大值在数组上的索引
        max_value_idx = idxs[last_idx]
        # 将这个添加到相应索引上
        pick.append(max_value_idx)

        xx1 = np.maximum(x1[max_value_idx], x1[idxs[: last_idx]])
        yy1 = np.maximum(y1[max_value_idx], y1[idxs[: last_idx]])
        xx2 = np.minimum(x2[max_value_idx], x2[idxs[: last_idx]])
        yy2 = np.minimum(y2[max_value_idx], y2[idxs[: last_idx]])

        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)

        iou = w * h / areas[idxs[: last_idx]]
        # 删除最大的value,并且删除iou > threshold的bounding boxes
        idxs = np.delete(idxs, np.concatenate(([last_idx], np.where(iou > threshold)[0])))

    # bounding box 返回一定要int类型,否则Opencv无法绘制
    return pick


class Yolov8TritonDetectionModel(DetectionModel):
    def __init__(self, model_path: str | None = None, model: Any | None = None, config_path: str | None = None, device: str | None = None, mask_threshold: float = 0.5, confidence_threshold: float = 0.3, category_mapping: Dict | None = None, category_remapping: Dict | None = None, load_at_init: bool = True, image_size: int = None,
                 tritonServer="", modelName="", modelVersion="", nms_threshold=0.5):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: str
                Torch device, "cpu" or "cuda"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        """
        super().__init__(model_path, model, config_path, device, mask_threshold, confidence_threshold, category_mapping, category_remapping, load_at_init, image_size)

        self.tritonServer = tritonServer
        self.modelName = modelName
        self.modelVersion = modelVersion
        self.nms_threshold = nms_threshold
        # Creat a random hash for request ID
        requestID = random.randint(0, 100000)
        self.requestID = sha256(str(requestID).encode('utf-8')).hexdigest() 

    
    def check_dependencies(self) -> None:
        check_requirements(["tritonclient", "turbojpeg"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        from turbojpeg import TurboJPEG
        import tritonclient.http as httpclient
        from tritonclient.utils import InferenceServerException

        try:
            # Create a HTTP client for inference server running in localhost:8000.
            self.triton_client = httpclient.InferenceServerClient(
                url=self.tritonServer,
            )
            self.jpeg = TurboJPEG() #TurboJPEG(r"C:\Program Files\libjpeg-turbo64\bin\turbojpeg.dll")
            self.modelName = self.modelName
            self.modelVersion = self.modelVersion
            print("triton_client.get_model_config(): ", self.triton_client.get_model_config(self.modelName, model_version=self.modelVersion)['input'])
            print("triton_client.get_model_metadata(): ", self.triton_client.get_model_metadata(self.modelName, model_version=self.modelVersion))

            self.input_w = self.image_size
            self.input_h = self.image_size

            self.httpclient = httpclient

            self.set_model(self.triton_client)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8 model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv8 model.
        Args:
            model: Any
                A YOLOv8 model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        
        inputs = []
        outputs = []

        scale = max(image.shape[0] / self.input_h, image.shape[1] / self.input_w)
        #ret, input_array = cv2.imencode('.jpg', image)[None]
        jpegData = self.jpeg.encode(image)
        input_array = np.frombuffer(jpegData, dtype = np.uint8)[None]

        #print("input_array: ", input_array)

        inputs.append(self.httpclient.InferInput('images', input_array.shape, "UINT8"))
        inputs[0].set_data_from_numpy(input_array)
        outputs.append(self.httpclient.InferRequestedOutput('output0'))
        

        # Send request to the inference server. Get results for both output tensors.
        try:
            resp = self.triton_client.async_infer(
                model_name=self.modelName,
                model_version=self.modelVersion,
                inputs=inputs,
                outputs=outputs,
                request_id=self.requestID
            )
            result = resp.get_result().as_numpy('output0').transpose((0, 2, 1))

            conf = np.max(result[0, :, 4:], axis=1)

            # Where the score larger than score_threshold
            mask = conf >= self.confidence_threshold
            result = result[:, mask, :]
            conf = conf[mask]

            clsid = np.argmax(result[0, :, 4:], axis=1)

            box = result[0, :, :4] * scale
            x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
            xmin = x - w / 2
            ymin = y - h / 2
            box4nms = np.zeros_like(box)
            box4nms[:, 0] = xmin + clsid * self.input_w
            box4nms[:, 1] = ymin + clsid * self.input_h
            box4nms[:, 2] = w
            box4nms[:, 3] = h

            # xywh2xyxy
            box4nms[:, 2] += box4nms[:, 0]
            box4nms[:, 3] += box4nms[:, 1]

            # Perform non-maximum suppression to eliminate redudant overlapping boxes with
            # lower confidences.
            keep_ids = nms(box4nms, conf, self.nms_threshold)

            print('box[keep_ids]: ', box[keep_ids])
            print('clsid[keep_ids]: ', clsid[keep_ids])
            print('conf[keep_ids]: ', conf[keep_ids])

            if box.size > 0:
                prediction_result = np.concatenate([box[keep_ids][None], clsid[keep_ids][:, None][None], conf[keep_ids][:, None][None]], axis=-1)
            else:
                prediction_result = np.array([[]])

        except Exception as e:
            print("unknown Exception: ", e)
            prediction_result = np.array([[]])

        print('prediction_result: ', prediction_result)
        self._original_predictions = prediction_result

    @property
    def category_names(self):
        return self.model.names.values()

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False  # fix when yolov5 supports segmentation models

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format:
                x = prediction[0]
                y = prediction[1]
                w = prediction[2]
                h = prediction[3]
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                bbox = [x1, y1, x2, y2]
                category_id = int(prediction[4])
                score = prediction[5]
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
