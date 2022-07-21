

from archipel.workers.worker import ImagesToImagesWorker

from .inference import get_model, predict 
from .inference_onnx import get_model_onnx, predict_onnx


__task_class_name__ = "FacePixelizer"


class FacePixelizer(ImagesToImagesWorker):
    def add_model_specific_args(self, parent_parser):
        parent_parser.add_argument(
            "--input-size",
            default=256,
            type=int,
            help="Network input size, imgs will resized in a square of this size",
        )
        parent_parser.add_argument("-l", "--local_model", help="If the model is local or from a url.", action='store_false')
        parent_parser.add_argument("-m", "--model_path", help="Path of the local model.", type=str)
        parent_parser.add_argument("-o", "--onnx", help="If the model type is onnx", action='store_flase')
        parent_parser.add_argument("-d", "--device", help="device to use (either cpu or cuda)", type=str, default="cpu")
        

    def setup_model(self):
        if self.args.onnx:
            self.model = get_model_onnx(self.args.local_model, self.args.model_path, self.args.input_size)
        else:
            self.model = get_model(self.args.local_model, self.args.model_path, self.args.input_size, device=self.args.device)

    def forward(self, imgs):
        if self.args.onnx:
            return predict(self.model, imgs[0])
        else:
            return predict_onnx(self.model, imgs[0])