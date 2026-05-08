"""
TensorFlow Lite model conversion, quantization, and C array export.
Convert full TF models to int8-quantized TFLite for MCU deployment.
SDKs: TensorFlow Lite, TFLite Runtime, NumPy
"""
import os
import struct
import numpy as np
from pathlib import Path
from typing import Optional, List, Callable, Tuple

import tensorflow as tf


class TFLiteConverter:
    """
    Convert and quantize TensorFlow models for embedded deployment.
    Targets: ESP32, Arduino Nano 33 BLE, Raspberry Pi Pico.
    """

    def __init__(self, model_or_path):
        """
        model_or_path: tf.keras.Model or path to SavedModel/H5/TFLite
        """
        if isinstance(model_or_path, str):
            self.model = tf.keras.models.load_model(model_or_path)
        else:
            self.model = model_or_path

    def convert_float32(self, output_path: str) -> str:
        """Convert to float32 TFLite — fast, larger model."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        size_kb = len(tflite_model) / 1024
        print(f"[TFLite] Float32 model: {output_path} ({size_kb:.1f}KB)")
        return output_path

    def convert_int8(
        self,
        output_path: str,
        representative_data: Optional[Callable] = None,
        input_shape: Optional[Tuple] = None,
        n_calibration_samples: int = 100,
    ) -> str:
        """
        Full int8 quantization for MCU deployment.
        representative_data: generator yielding (1, ...) float32 tensors for calibration.
        Falls back to random calibration data if not provided.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        if representative_data is None:
            # Auto-generate calibration data from model input shape
            if input_shape is None:
                input_shape = self.model.input_shape[1:]  # drop batch dim
            def _default_rep_data():
                for _ in range(n_calibration_samples):
                    sample = np.random.normal(0, 1, (1,) + tuple(input_shape)).astype(np.float32)
                    yield [sample]
            representative_data = _default_rep_data

        converter.representative_dataset = representative_data
        tflite_model = converter.convert()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        size_kb = len(tflite_model) / 1024
        print(f"[TFLite] Int8 quantized: {output_path} ({size_kb:.1f}KB)")
        return output_path

    def to_c_array(self, tflite_path: str, output_path: Optional[str] = None, var_name: str = "model_data") -> str:
        """
        Convert .tflite binary to C header file for embedding in firmware.
        Output: model_data.h with uint8_t array and length constant.
        """
        with open(tflite_path, "rb") as f:
            model_bytes = f.read()

        output_path = output_path or tflite_path.replace(".tflite", ".h")
        array_len = len(model_bytes)

        hex_vals = ", ".join(f"0x{b:02x}" for b in model_bytes)
        lines = [f"// Auto-generated from {Path(tflite_path).name}",
                 f"// Size: {array_len} bytes ({array_len/1024:.1f}KB)",
                 f"",
                 f"#ifndef {var_name.upper()}_H",
                 f"#define {var_name.upper()}_H",
                 f"",
                 f"#include <stdint.h>",
                 f"",
                 f"const uint8_t {var_name}[] = {{",
                 f"  {hex_vals}",
                 f"}};",
                 f"const unsigned int {var_name}_len = {array_len};",
                 f"",
                 f"#endif  // {var_name.upper()}_H",
                 ]

        with open(output_path, "w") as f:
            f.write("
".join(lines))
        print(f"[TFLite] C header: {output_path} ({array_len} bytes)")
        return output_path

    def benchmark(self, tflite_path: str, n_runs: int = 100) -> dict:
        """Benchmark TFLite model inference latency on host CPU."""
        import time
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()
        out = interpreter.get_output_details()

        input_shape = inp[0]["shape"]
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(5):
            interpreter.set_tensor(inp[0]["index"], dummy_input)
            interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            interpreter.set_tensor(inp[0]["index"], dummy_input)
            interpreter.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        result = {
            "model": tflite_path,
            "input_shape": input_shape.tolist(),
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "n_runs": n_runs,
        }
        print(f"[TFLite] Benchmark: mean={result['mean_ms']:.2f}ms, p95={result['p95_ms']:.2f}ms")
        return result


def build_anomaly_detector(input_length: int = 100, n_features: int = 1) -> tf.keras.Model:
    """
    Build a compact 1D CNN anomaly detector for vibration/sensor data.
    Designed to fit on ESP32 (< 100KB after int8 quantization).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_length, n_features)),
        tf.keras.layers.Conv1D(8, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),  # normal / anomaly
    ], name="vibration_anomaly_detector")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    total_params = model.count_params()
    print(f"[TFLite] Model: {total_params:,} params")
    model.summary()
    return model
