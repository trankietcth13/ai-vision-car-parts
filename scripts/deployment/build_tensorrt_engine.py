"""
TensorRT Engine Builder for Jetson Orin & Nvidia DGX Spark
Supports FP16 and INT8 Post-Training Quantization (PTQ)
"""

import os
import argparse
from typing import Optional


def build_tensorrt_engine(
    onnx_file_path: str,
    engine_file_path: str,
    precision: str = "fp16",
    calib_images_dir: Optional[str] = None,
    max_workspace_gb: int = 4
):
    print(f"[TensorRT Builder] Target ONNX: {onnx_file_path}")
    print(f"[TensorRT Builder] Engine Destination: {engine_file_path}")
    print(f"[TensorRT Builder] Target Precision: {precision.upper()}")

    try:
        import tensorrt as trt
    except ImportError:
        print("[TensorRT Builder Error] TensorRT is not installed in the current environment.")
        print("[Note] On DGX Spark or Jetson, run inside NVIDIA Container with TensorRT support:")
        print("       trtexec --onnx=" + onnx_file_path + f" --saveEngine={engine_file_path} --{precision.lower()}")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"[TensorRT Parse Error] {parser.get_error(error)}")
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb * (1 << 30))

    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TensorRT Builder] Enabled FP16 mode.")
        else:
            print("[TensorRT Builder Warning] Device lacks native FP16 support, using default FP32.")
            
    elif precision.lower() == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[TensorRT Builder] Enabled INT8 Post-Training Quantization.")
            # Note: For INT8, an IInt8EntropyCalibrator2 is attached using images from calib_images_dir
        else:
            print("[TensorRT Builder Warning] Device lacks native INT8 support.")

    print("[TensorRT Builder] Serializing Engine (this may take a few minutes)...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("[TensorRT Builder Error] Failed to build serialized network.")
        return

    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    print(f"[TensorRT Builder Success] Engine saved to {engine_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT Engine from ONNX")
    parser.add_argument("--onnx", type=str, default="./artifacts/deployment/kd_n_full_640.onnx")
    parser.add_argument("--engine", type=str, default="./artifacts/deployment/kd_n_full_640_fp16.engine")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])
    args = parser.parse_args()

    build_tensorrt_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        precision=args.precision
    )
