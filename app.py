import argparse
import logging
from typing import Union
import requests
from PIL import Image

# Import necessary modules/classes for Clams and MMIF
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

# Import PyTorch, DETR-related libraries, and transformers
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

class DetrObjectDetectionWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        if torch.cuda.is_available():
            self.model.to('cuda')

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        self.logger.debug("running app")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)

        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
            self.logger.debug(timeframe.properties)
            representatives = timeframe.get("representatives") if "representatives" in timeframe.properties else None
            if representatives:
                representative: AnnotationTypes.TimePoint = input_view.get_annotation_by_id(representatives[0])
                self.logger.debug("Sampling 1 frame")
                rep_frame = vdh.convert(representative.get("timePoint"), "milliseconds",
                                        "frame", vdh.get_framerate(video_doc))
                timepoint = representative
            else:
                self.logger.debug("No representatives, using middle frame")
                start_time = timeframe.get("start")
                end_time = timeframe.get("end")
                middle_time = (start_time + end_time) / 2
                rep_frame = vdh.convert(middle_time, "milliseconds", "frame", vdh.get_framerate(video_doc))
                # Create a new TimePoint annotation for the middle frame
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property('timePoint', middle_time)

            image = vdh.extract_frames_as_images(video_doc, [rep_frame], as_PIL=True)[0]
            self.logger.debug("Extracted image for object detection")

            inputs = self.processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = self.model(**inputs)
            self.logger.debug("Object detection completed")

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.5:  # Adjust threshold as needed
                    box = [round(i, 2) for i in box.tolist()]
                    self.logger.debug(
                        f"Detected {self.model.config.id2label[label.item()]} with confidence "
                        f"{round(score.item(), 3)} at location {box}"
                    )
                    bbox_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                    bbox_annotation.add_property("coordinates", box)
                    bbox_annotation.add_property("label", self.model.config.id2label[label.item()])
                    bbox_annotation.add_property("confidence", round(score.item(), 3))  # Add confidence property
                    time_bbox_alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                    time_bbox_alignment.add_property("source", timepoint.id)
                    time_bbox_alignment.add_property("target", bbox_annotation.id)

        return mmif

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = DetrObjectDetectionWrapper()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()


