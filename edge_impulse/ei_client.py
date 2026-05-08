"""
Edge Impulse SDK client for TinyML development cycle.
Data upload, impulse training, model export, and deployment.
SDKs: Edge Impulse SDK, httpx, numpy
"""
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import httpx

EI_API_BASE = "https://studio.edgeimpulse.com/v1"


@dataclass
class SensorSample:
    label: str
    values: List[float]       # raw sensor readings (accelerometer, etc.)
    sample_rate_hz: int = 100
    sensor_name: str = "accelerometer"
    interval_ms: float = 10.0


class EdgeImpulseClient:
    """
    Edge Impulse API client: manage datasets, train impulses, export models.
    Full TinyML development cycle from data to deployed .tflite.
    """

    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EI_API_KEY", "")
        self.project_id = project_id or os.environ.get("EI_PROJECT_ID", "")
        if not self.api_key:
            raise ValueError("EI_API_KEY required. Set env var or pass api_key=")
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        self.client = httpx.Client(timeout=60.0)
        print(f"[EdgeImpulse] Client initialized for project {self.project_id}")

    def upload_sample(self, sample: SensorSample, category: str = "training") -> dict:
        """
        Upload a labeled sensor sample to Edge Impulse dataset.
        category: 'training', 'testing', or 'anomaly'
        """
        payload = {
            "protected": {
                "ver": "v1",
                "alg": "none",
                "iat": int(time.time()),
            },
            "signature": "0000",
            "payload": {
                "device_name": "local_simulator",
                "device_type": "ESP32",
                "interval_ms": sample.interval_ms,
                "sensors": [{"name": sample.sensor_name, "units": "m/s2"}],
                "values": [[v] for v in sample.values],
            }
        }
        url = f"{EI_API_BASE}/api/{self.project_id}/training/{category}/data"
        response = self.client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def upload_batch(
        self,
        samples: List[SensorSample],
        category: str = "training",
        verbose: bool = True,
    ) -> List[dict]:
        """Upload multiple samples. Returns list of upload results."""
        results = []
        for i, sample in enumerate(samples):
            try:
                result = self.upload_sample(sample, category)
                results.append(result)
                if verbose and (i + 1) % 10 == 0:
                    print(f"[EI] Uploaded {i+1}/{len(samples)} samples")
            except Exception as e:
                print(f"[EI] Upload failed for sample {i}: {e}")
                results.append({"error": str(e)})
        print(f"[EI] Batch upload complete: {len(results)} samples")
        return results

    def start_training(self, timeout_minutes: int = 10) -> dict:
        """Trigger model training and poll until complete."""
        url = f"{EI_API_BASE}/api/{self.project_id}/jobs/train/nn"
        resp = self.client.post(url, headers=self.headers)
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        print(f"[EI] Training started. Job ID: {job_id}")

        start = time.time()
        while time.time() - start < timeout_minutes * 60:
            status = self.get_job_status(job_id)
            state = status.get("job", {}).get("finishedSuccessful")
            if state is True:
                print(f"[EI] Training complete!")
                return status
            elif state is False:
                print(f"[EI] Training failed: {status}")
                return status
            print(f"[EI] Training... ({int(time.time()-start)}s elapsed)")
            time.sleep(10)

        print("[EI] Training timed out")
        return {}

    def get_job_status(self, job_id: str) -> dict:
        url = f"{EI_API_BASE}/api/{self.project_id}/jobs/{job_id}/status"
        resp = self.client.get(url, headers=self.headers)
        return resp.json()

    def export_tflite(self, output_path: str = "./models/model.tflite") -> str:
        """Export trained model as TFLite file."""
        url = f"{EI_API_BASE}/api/{self.project_id}/deployment/download?type=tflite"
        resp = self.client.get(url, headers=self.headers)
        resp.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"[EI] TFLite model exported: {output_path} ({len(resp.content)/1024:.1f}KB)")
        return output_path

    def export_arduino_library(self, output_path: str = "./firmware/ei_library.zip") -> str:
        """Export model as Arduino library .zip for direct firmware inclusion."""
        url = f"{EI_API_BASE}/api/{self.project_id}/deployment/download?type=arduino"
        resp = self.client.get(url, headers=self.headers)
        resp.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"[EI] Arduino library exported: {output_path}")
        return output_path

    def get_model_accuracy(self) -> dict:
        """Fetch latest training performance metrics."""
        url = f"{EI_API_BASE}/api/{self.project_id}/learn/8/accuracy"
        resp = self.client.get(url, headers=self.headers)
        return resp.json()


def generate_vibration_samples(
    n_normal: int = 200,
    n_anomaly: int = 50,
    sample_length: int = 100,
    sample_rate_hz: int = 100,
    seed: int = 42,
) -> Tuple[List[SensorSample], List[SensorSample]]:
    """
    Generate synthetic vibration sensor data for anomaly detection training.
    Normal: low-amplitude oscillation. Anomaly: high-frequency bursts.
    """
    rng = np.random.default_rng(seed)
    interval_ms = 1000.0 / sample_rate_hz

    normal_samples = []
    for i in range(n_normal):
        t = np.linspace(0, sample_length / sample_rate_hz, sample_length)
        freq = rng.uniform(10, 30)
        amp = rng.uniform(0.2, 0.8)
        noise = rng.normal(0, 0.05, sample_length)
        values = (amp * np.sin(2 * np.pi * freq * t) + noise).tolist()
        normal_samples.append(SensorSample(
            label="normal", values=values,
            sample_rate_hz=sample_rate_hz, interval_ms=interval_ms
        ))

    anomaly_samples = []
    for i in range(n_anomaly):
        t = np.linspace(0, sample_length / sample_rate_hz, sample_length)
        # Anomaly: sudden high-freq burst + DC offset
        burst_start = rng.integers(20, 70)
        values = rng.normal(0, 0.1, sample_length)
        values[burst_start:burst_start+20] += rng.uniform(2.0, 5.0) * np.sin(
            2 * np.pi * rng.uniform(80, 150) * t[burst_start:burst_start+20]
        )
        anomaly_samples.append(SensorSample(
            label="anomaly", values=values.tolist(),
            sample_rate_hz=sample_rate_hz, interval_ms=interval_ms
        ))

    print(f"[EI] Generated {n_normal} normal + {n_anomaly} anomaly samples")
    return normal_samples, anomaly_samples
