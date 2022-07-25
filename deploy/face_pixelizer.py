

from archipel.workers.worker import ImagesToImagesWorker

from inference import get_model, predict


__task_class_name__ = "FacePixelizer"


class FacePixelizer(ImagesToImagesWorker):
    def add_model_specific_args(self, parent_parser):
        parent_parser.add_argument(
            "--input-size",
            default=256,
            type=int,
            help="Network input size, imgs will resized in a square of this size",
        )
        parent_parser.add_argument("-m", "--model_path", help="Path of the local model.", type=str)
        

    def setup_model(self):
        self.model = get_model(self.args.model_path, self.args.input_size)

    def forward(self, imgs):
        return [predict(self.model, img) for img in imgs]
