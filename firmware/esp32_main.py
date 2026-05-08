"""
ESP32 MicroPython firmware: vibration sampling + TFLite inference + MQTT telemetry.
Deploy this file to ESP32 as main.py via ampy, rshell, or Thonny.
SDKs: MicroPython, TFLite (via ulab), MQTT
"""
# NOTE: This runs on MicroPython on ESP32.
# Host Python tools (ampy, esptool) are used for flashing, not this file itself.

_FIRMWARE_SOURCE = """
import machine
import network
import time
import json
import uos
from ulab import numpy as np

# ---- Config ----
WIFI_SSID = "YOUR_SSID"
WIFI_PASS = "YOUR_PASSWORD"
MQTT_BROKER = "192.168.1.100"
MQTT_PORT = 1883
MQTT_TOPIC = b"sensor/esp32/vibration"
DEVICE_ID = "esp32_node_01"
SAMPLE_RATE_HZ = 100
SAMPLE_LENGTH = 100

# ---- Hardware ----
# Accelerometer on I2C (MPU6050 or similar)
i2c = machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21), freq=400000)

def read_accel_mpu6050():
    MPU_ADDR = 0x68
    # Wake up MPU6050
    i2c.writeto_mem(MPU_ADDR, 0x6B, b"\x00")
    # Read 6 bytes from ACCEL_XOUT_H
    raw = i2c.readfrom_mem(MPU_ADDR, 0x3B, 6)
    def to_signed(h, l):
        val = (h << 8) | l
        return val - 65536 if val > 32767 else val
    ax = to_signed(raw[0], raw[1]) / 16384.0  # g
    ay = to_signed(raw[2], raw[3]) / 16384.0
    az = to_signed(raw[4], raw[5]) / 16384.0
    return ax, ay, az

# ---- WiFi ----
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(WIFI_SSID, WIFI_PASS)
        for _ in range(20):
            if wlan.isconnected():
                break
            time.sleep(0.5)
    if wlan.isconnected():
        print("WiFi:", wlan.ifconfig()[0])
        return True
    print("WiFi FAILED")
    return False

# ---- MQTT ----
def mqtt_publish(client, topic, payload_dict):
    msg = json.dumps(payload_dict).encode()
    client.publish(topic, msg)

# ---- TFLite inference (via tflite-micro binding or stub) ----
try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_AVAILABLE = True
except ImportError:
    _TFLITE_AVAILABLE = False

def run_inference(sample_values):
    if not _TFLITE_AVAILABLE:
        # Stub: simple threshold anomaly
        rms = sum(v*v for v in sample_values) ** 0.5 / len(sample_values) ** 0.5
        return "anomaly" if rms > 1.5 else "normal", float(rms)
    try:
        import uos
        interpreter = tflite.Interpreter(model_path="/model.tflite")
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        arr = np.array(sample_values, dtype=np.float32).reshape(inp["shape"])
        interpreter.set_tensor(inp["index"], arr)
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"])[0]
        label = "anomaly" if probs[1] > probs[0] else "normal"
        return label, float(max(probs))
    except Exception as e:
        return "error", 0.0

# ---- Main loop ----
def main():
    connect_wifi()

    try:
        from umqtt.simple import MQTTClient
        client = MQTTClient(DEVICE_ID, MQTT_BROKER, port=MQTT_PORT)
        client.connect()
        print("MQTT connected")
    except Exception as e:
        print("MQTT failed:", e)
        client = None

    interval_us = int(1_000_000 / SAMPLE_RATE_HZ)
    sample = []

    while True:
        ax, ay, az = read_accel_mpu6050()
        magnitude = (ax*ax + ay*ay + az*az) ** 0.5
        sample.append(magnitude)

        if len(sample) >= SAMPLE_LENGTH:
            label, confidence = run_inference(sample)
            payload = {
                "device": DEVICE_ID,
                "ts": time.time(),
                "label": label,
                "confidence": confidence,
                "rms": sum(v*v for v in sample)**0.5 / len(sample)**0.5,
            }
            print(payload)
            if client:
                try:
                    mqtt_publish(client, MQTT_TOPIC, payload)
                except Exception:
                    pass
            sample = []

        time.sleep_us(interval_us)

main()
"""

# This module exposes the firmware source as a string for flashing tools.
# Use flash_firmware() to write main.py to the device via ampy.

def flash_firmware(
    device_port: str = "/dev/ttyUSB0",
    baud: int = 115200,
    output_path: str = "/tmp/esp32_main.py",
):
    """Write firmware to ESP32 via ampy (ampy --port /dev/ttyUSB0 put main.py)."""
    import subprocess, tempfile, os
    with open(output_path, "w") as f:
        f.write(_FIRMWARE_SOURCE)
    print(f"[Firmware] Written to {output_path}")

    cmd = ["ampy", "--port", device_port, "--baud", str(baud), "put", output_path, "/main.py"]
    print(f"[Firmware] Flashing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print("[Firmware] Flash successful")
    else:
        print(f"[Firmware] Flash failed: {result.stderr}")
    return result.returncode == 0
