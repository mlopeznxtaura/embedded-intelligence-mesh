"""
embedded-intelligence-mesh — Entry Point

TinyML fleet: train anomaly detector, convert to TFLite, flash ESP32,
simulate mesh, ingest MQTT telemetry, manage OTA updates.

Usage:
  python main.py --mode train --output ./models
  python main.py --mode convert --model ./models/anomaly_detector.h5
  python main.py --mode simulate --nodes 10 --steps 200 --broker localhost
  python main.py --mode monitor --broker localhost --port 1883
  python main.py --mode pipeline  (full end-to-end)
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Embedded Intelligence Mesh")
    parser.add_argument("--mode", required=True,
                        choices=["train", "convert", "simulate", "monitor", "pipeline"])
    parser.add_argument("--output", default="./output")
    parser.add_argument("--model", default=None, help="Path to .h5 or .tflite model")
    parser.add_argument("--nodes", type=int, default=10, help="Virtual nodes (simulate mode)")
    parser.add_argument("--steps", type=int, default=500, help="Steps per node (simulate mode)")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anomaly-rate", type=float, default=0.1)
    return parser.parse_args()


def mode_train(args):
    """Train vibration anomaly detector, save as .h5 and .tflite."""
    import numpy as np
    import tensorflow as tf
    from edge_impulse.ei_client import generate_vibration_samples
    from tflite.model_converter import TFLiteConverter, build_anomaly_detector

    Path(args.output).mkdir(parents=True, exist_ok=True)
    normal, anomaly = generate_vibration_samples(
        n_normal=300, n_anomaly=100, seed=args.seed
    )

    # Build dataset
    X = np.array([s.values for s in normal + anomaly], dtype=np.float32)
    y = np.array([0] * len(normal) + [1] * len(anomaly), dtype=np.int32)
    X = X[:, :, np.newaxis]  # (N, 100, 1)

    idx = np.random.default_rng(args.seed).permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_anomaly_detector(input_length=100, n_features=1)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=20, batch_size=32, verbose=1)

    h5_path = str(Path(args.output) / "anomaly_detector.h5")
    model.save(h5_path)
    print(f"
Model saved: {h5_path}")

    # Convert to TFLite
    converter = TFLiteConverter(model)
    tflite_path = str(Path(args.output) / "anomaly_detector.tflite")
    int8_path = str(Path(args.output) / "anomaly_detector_int8.tflite")
    converter.convert_float32(tflite_path)
    converter.convert_int8(int8_path, input_shape=(100, 1))

    # Export C header
    converter.to_c_array(int8_path, str(Path(args.output) / "model_data.h"))
    bench = converter.benchmark(tflite_path)
    print(f"Benchmark: {bench['mean_ms']:.2f}ms mean inference")


def mode_convert(args):
    """Convert existing model to TFLite + C header."""
    if not args.model:
        print("--model required"); sys.exit(1)
    from tflite.model_converter import TFLiteConverter
    converter = TFLiteConverter(args.model)
    out = str(Path(args.output) / "model_int8.tflite")
    converter.convert_int8(out)
    converter.to_c_array(out, str(Path(args.output) / "model_data.h"))


def mode_simulate(args):
    """Simulate N virtual ESP32 nodes publishing to MQTT."""
    from simulation.mesh_simulator import MeshSimulator
    sim = MeshSimulator(
        n_nodes=args.nodes,
        broker=args.broker,
        port=args.port,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )
    print(f"Simulating {args.nodes} nodes x {args.steps} steps...")
    sim._start_time = __import__("time").time()
    sim.run(steps=args.steps)


def mode_monitor(args):
    """Listen to MQTT and log all sensor readings."""
    from mqtt.telemetry import MQTTTelemetryPipeline
    from prometheus_client import start_http_server

    start_http_server(9090)
    print("Prometheus metrics: http://localhost:9090/metrics")

    def on_anomaly(reading):
        print(f"ALERT: {reading.device} anomaly (rms={reading.rms:.3f})")

    pipeline = MQTTTelemetryPipeline(
        broker=args.broker,
        port=args.port,
        on_anomaly=on_anomaly,
    )
    pipeline.start(block=True)


def mode_pipeline(args):
    """Full pipeline: train -> convert -> simulate -> monitor (demo run)."""
    print("Running full pipeline demo...")
    args.output = args.output or "./pipeline_output"
    mode_train(args)
    mode_simulate(args)


def main():
    args = parse_args()
    print("=" * 60)
    print("  Embedded Intelligence Mesh")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 60)

    dispatch = {
        "train": mode_train,
        "convert": mode_convert,
        "simulate": mode_simulate,
        "monitor": mode_monitor,
        "pipeline": mode_pipeline,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
