from archipel.workers.worker import ImagesToImagesWorker

from face_pixelizer import FacePixelizer


__task_class_name__ = "ArchipelFacePixelizer"


class ArchipelFacePixelizer(ImagesToImagesWorker):
    def add_model_specific_args(self, parent_parser):
        parent_parser.add_argument(
            "--input-size",
            default=1080,
            type=int,
            help="Network input size, imgs will resized in a square of this size",
        )
        parent_parser.add_argument(
            "--score-threshold",
            default=0.4,
            type=float,
            help="Discords all results with confidence score < score-threshold",
        )
        parent_parser.add_argument(
            "--nms-threshold",
            default=0.4,
            type=float,
            help="Discards all overlapping boxes with IoU > nms-threshold",
        )
        parent_parser.add_argument(
            "--state-dict",
            default="/opt/face_pixelizer/retinaface_mobilenet_0.25.pth",
            type=str,
            help="Path to pretrained weights",
        )

    def setup_model(self):
        self.model = FacePixelizer(
            self.args.input_size,
            self.args.score_threshold,
            self.args.nms_threshold,
            self.args.state_dict,
        )

    def forward(self, imgs):
        return self.model(imgs)
