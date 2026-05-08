# Embedded Intelligence Mesh

Cluster 11 of the NextAura 500 SDKs / 25 Clusters project.

Fleet of microcontrollers running TinyML with OTA learning and mesh coordination. Edge Impulse development cycle for ESP32 vibration anomaly detection — data collection, training, export, on-device inference.

## Architecture

- Edge Impulse SDK for TinyML model training and deployment
- TensorFlow Lite Micro for on-device inference
- ESP-IDF / Arduino SDK for ESP32 firmware
- Zephyr RTOS for multi-threaded embedded scheduling
- FreeRTOS for lightweight task management
- MicroPython / CircuitPython for rapid prototyping
- Mosquitto MQTT for sensor data telemetry
- CoAP for constrained device messaging
- Balena for OTA fleet management and updates
- AWS IoT for cloud device management and shadow state
- TimescaleDB for time-series sensor data
- Grafana for real-time sensor dashboards
- Prometheus for metrics collection

## SDKs Used

Edge Impulse SDK, TensorFlow Lite Micro, Arduino SDK, ESP-IDF, Zephyr RTOS SDK, MicroPython SDK, CircuitPython SDK, Mosquitto MQTT, CoAP SDK, Nordic nRF SDK, Arm CMSIS SDK, FreeRTOS SDK, Balena SDK, AWS IoT SDK, Prometheus Client, TimescaleDB, Grafana SDK, FastAPI, Redis SDK, Nix SDK

## Quickstart

```bash
pip install -r requirements.txt

# Train anomaly detector and deploy to ESP32
python main.py --mode train --data ./sensor_data --label vibration_anomaly
python main.py --mode deploy --model ./models/anomaly_detector.tflite --device /dev/ttyUSB0
python main.py --mode monitor --broker localhost:1883

# Simulate a full mesh of 10 virtual ESP32 nodes
python main.py --mode simulate --nodes 10 --steps 500
```

## Structure

```
edge_impulse/    Edge Impulse API client, dataset upload, model training
tflite/          TFLite Micro model conversion, quantization, C array export
firmware/        ESP-IDF / Arduino / MicroPython device firmware templates
mqtt/            Mosquitto MQTT telemetry pipeline
coap/            CoAP constrained messaging for low-power devices
ota/             Balena + AWS IoT OTA update orchestration
timeseries/      TimescaleDB ingestion + Grafana dashboard provisioning
simulation/      Virtual ESP32 mesh simulation for local dev/testing
api/             FastAPI device management backend
main.py          Entry point
```
