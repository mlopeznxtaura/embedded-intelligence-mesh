"""
MQTT telemetry pipeline for sensor data collection.
Subscribes to device topics, stores to TimescaleDB, publishes alerts.
SDKs: paho-mqtt, psycopg2, Redis, Prometheus Client
"""
import json
import time
import threading
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from prometheus_client import Counter, Gauge, Histogram

# Prometheus metrics
MQTT_MSGS_RECEIVED = Counter("mqtt_messages_received_total", "Total MQTT messages received", ["topic"])
MQTT_ANOMALIES = Counter("mqtt_anomalies_total", "Total anomaly detections", ["device"])
SENSOR_RMS = Gauge("sensor_rms_value", "Latest sensor RMS value", ["device"])
INFERENCE_CONFIDENCE = Histogram(
    "inference_confidence", "TFLite inference confidence scores",
    ["device", "label"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)


@dataclass
class SensorReading:
    device: str
    timestamp: float
    label: str           # "normal" or "anomaly"
    confidence: float
    rms: float
    raw_payload: dict


class MQTTTelemetryPipeline:
    """
    Subscribe to sensor MQTT topics, parse readings, store to DB, fire alerts.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topics: Optional[List[str]] = None,
        on_anomaly: Optional[Callable[[SensorReading], None]] = None,
        db_conn_str: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        self.broker = broker
        self.port = port
        self.topics = topics or ["sensor/+/vibration", "sensor/+/temperature", "sensor/+/status"]
        self.on_anomaly = on_anomaly
        self.readings: List[SensorReading] = []
        self._running = False
        self._db = None
        self._redis = None

        # DB connection (optional)
        if db_conn_str:
            try:
                import psycopg2
                self._db = psycopg2.connect(db_conn_str)
                self._setup_timescale()
                print("[MQTT] TimescaleDB connected")
            except Exception as e:
                print(f"[MQTT] DB unavailable: {e}")

        # Redis (optional)
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                print("[MQTT] Redis connected")
            except Exception as e:
                print(f"[MQTT] Redis unavailable: {e}")

        # MQTT client
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _setup_timescale(self):
        """Create hypertable for sensor readings if not exists."""
        with self._db.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    time TIMESTAMPTZ NOT NULL,
                    device TEXT,
                    label TEXT,
                    confidence FLOAT,
                    rms FLOAT
                )
            """)
            try:
                cur.execute("SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE)")
            except Exception:
                pass  # Already a hypertable or TimescaleDB not installed
            self._db.commit()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"[MQTT] Connected to {self.broker}:{self.port} (rc={reason_code})")
        for topic in self.topics:
            client.subscribe(topic)
            print(f"[MQTT] Subscribed: {topic}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        print(f"[MQTT] Disconnected (rc={reason_code}). Reconnecting...")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            reading = SensorReading(
                device=payload.get("device", "unknown"),
                timestamp=payload.get("ts", time.time()),
                label=payload.get("label", "unknown"),
                confidence=float(payload.get("confidence", 0)),
                rms=float(payload.get("rms", 0)),
                raw_payload=payload,
            )
            self.readings.append(reading)
            MQTT_MSGS_RECEIVED.labels(topic=msg.topic).inc()
            SENSOR_RMS.labels(device=reading.device).set(reading.rms)
            INFERENCE_CONFIDENCE.labels(device=reading.device, label=reading.label).observe(reading.confidence)

            if reading.label == "anomaly":
                MQTT_ANOMALIES.labels(device=reading.device).inc()
                print(f"[MQTT] ANOMALY from {reading.device}: rms={reading.rms:.3f}, conf={reading.confidence:.3f}")
                if self.on_anomaly:
                    self.on_anomaly(reading)

            self._store_reading(reading)
            self._update_redis(reading)

        except Exception as e:
            print(f"[MQTT] Parse error: {e} | payload: {msg.payload[:100]}")

    def _store_reading(self, reading: SensorReading):
        if self._db:
            try:
                with self._db.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sensor_readings VALUES (%s, %s, %s, %s, %s)",
                        (datetime.fromtimestamp(reading.timestamp, tz=timezone.utc),
                         reading.device, reading.label, reading.confidence, reading.rms)
                    )
                self._db.commit()
            except Exception as e:
                print(f"[MQTT] DB write error: {e}")

    def _update_redis(self, reading: SensorReading):
        if self._redis:
            try:
                self._redis.hset(
                    f"device:{reading.device}:latest",
                    mapping={
                        "label": reading.label,
                        "confidence": reading.confidence,
                        "rms": reading.rms,
                        "ts": reading.timestamp,
                    }
                )
                self._redis.expire(f"device:{reading.device}:latest", 300)
            except Exception:
                pass

    def start(self, block: bool = True):
        self.client.connect(self.broker, self.port, keepalive=60)
        self._running = True
        if block:
            print(f"[MQTT] Listening on {self.broker}:{self.port}...")
            self.client.loop_forever()
        else:
            self.client.loop_start()

    def stop(self):
        self._running = False
        self.client.loop_stop()
        self.client.disconnect()
        if self._db:
            self._db.close()
