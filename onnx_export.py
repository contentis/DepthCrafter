import torch
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from argparse import ArgumentParser
import os
import onnx
import shutil

@torch.inference_mode()
def onnx_exporter(unet, args):
    tmp_dir = "tmp"
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    os.makedirs(tmp_dir, exist_ok=True)
    onnx_tmp = os.path.join(tmp_dir, "backbone.onnx")

    model_inputs = {
    "sample": (1,"num_franes",8,"h/8","w/8"),
    "timestep": (1,),
    "encoder_hidden_states": (1, "num_franes", 1024),
    "added_time_ids": (1,3)
    }

    input_names = ["sample", "timestep", "encoder_hidden_states", "added_time_ids"]
    output_names = ["latent"]
    inputs = (
        torch.zeros(1, args.n_frames, 8, args.height//8, args.width//8).to(dtype=torch.half, device="cuda"),
        torch.zeros(1,).to(dtype=torch.half, device="cuda"),
        torch.zeros(1, args.n_frames, 1024).to(dtype=torch.half, device="cuda"),
        torch.zeros(1, 3).to(dtype=torch.half, device="cuda"),
    )

    dyn_axes = {
        "sample": {1: "num_frames", 3: "h", 4:"w"},
        "encoder_hidden_states": {1: "num_frames"}
    }

    torch.onnx.export(
        unet,
        inputs,
        onnx_tmp,
        dynamic_axes=dyn_axes,
        opset_version=17,
        input_names=input_names,
        output_names=output_names
    )
    onnx_model = onnx.load(onnx_tmp, load_external_data=True)
    onnx.save(
        onnx_model,
        os.path.join(args.o, "backbone.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="backbone.onnx" + "_data",
        size_threshold=1024,
    )
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--unet-path",
        type=str,
        default="tencent/DepthCrafter",
        help="Path to the UNet model",
    )
    parser.add_argument("-width", type=int, default=512)
    parser.add_argument("-height", type=int, default=512)
    parser.add_argument("-n_frames", type=int, default=8)
    parser.add_argument("-o", type=str, default="onnx")
    args = parser.parse_args()



    unet_path = 'tencent/DepthCrafter'
    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                unet_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )

    unet.to("cuda")
    onnx_exporter(unet, args)
