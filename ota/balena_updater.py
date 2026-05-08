"""
Balena + AWS IoT OTA update orchestration.
Push firmware updates to entire device fleet with rollback support.
SDKs: Balena SDK, AWS IoT SDK (boto3)
"""
import os
import json
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import balena
    BALENA_AVAILABLE = True
except ImportError:
    BALENA_AVAILABLE = False
    print("Warning: balena-sdk not available. Install: pip install balena-sdk")

import boto3


class BalenaFleetUpdater:
    """
    Manage OTA updates to ESP32 / Raspberry Pi fleet via Balena Cloud.
    Supports rolling updates, environment variable injection, and device tags.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BALENA_API_KEY", "")
        if not BALENA_AVAILABLE:
            raise ImportError("balena-sdk required. Install: pip install balena-sdk")
        self.sdk = balena.Balena()
        if self.api_key:
            self.sdk.auth.login_with_token(self.api_key)
        print("[Balena] SDK initialized")

    def list_devices(self, fleet_id: str) -> List[Dict]:
        """List all devices in a fleet with their status."""
        devices = self.sdk.models.device.get_all_by_application(fleet_id)
        for d in devices:
            print(f"  {d['device_name']:30s} | {d.get('status', 'unknown'):10s} | "
                  f"online: {d.get('is_online', False)}")
        return devices

    def push_env_var(self, fleet_id: str, key: str, value: str):
        """Set a fleet-wide environment variable (e.g. MODEL_VERSION)."""
        self.sdk.models.environment_variables.application.create(fleet_id, key, value)
        print(f"[Balena] Set {key}={value} on fleet {fleet_id}")

    def pin_release(self, fleet_id: str, release_id: str):
        """Pin fleet to a specific release (rollback-safe)."""
        self.sdk.models.application.pin_to_release(fleet_id, release_id)
        print(f"[Balena] Fleet {fleet_id} pinned to release {release_id}")

    def restart_fleet(self, fleet_id: str):
        """Restart all services on all devices in fleet."""
        devices = self.sdk.models.device.get_all_by_application(fleet_id)
        for device in devices:
            if device.get("is_online"):
                self.sdk.models.device.restart_service(device["id"])
        print(f"[Balena] Restarted {len(devices)} devices")

    def get_device_logs(self, device_uuid: str, count: int = 100) -> List[str]:
        """Fetch recent logs from a specific device."""
        logs = self.sdk.logs.history(device_uuid, count=count)
        return [f"[{l.get('timestamp')}] {l.get('message', '')}" for l in logs]


class AWSIoTFleetManager:
    """
    Manage device fleet, shadow state, and OTA jobs via AWS IoT Core.
    SDKs: boto3 (AWS IoT, S3, IoT Jobs)
    """

    def __init__(self, region: str = "us-east-1"):
        self.iot = boto3.client("iot", region_name=region)
        self.iot_data = boto3.client("iot-data", region_name=region)
        self.s3 = boto3.client("s3", region_name=region)
        print(f"[AWS IoT] Client initialized in {region}")

    def list_things(self, thing_type: str = "ESP32") -> List[Dict]:
        """List registered IoT things."""
        resp = self.iot.list_things(thingTypeName=thing_type)
        things = resp.get("things", [])
        print(f"[AWS IoT] {len(things)} devices of type {thing_type}")
        return things

    def get_shadow(self, thing_name: str) -> Dict:
        """Get device shadow (desired + reported state)."""
        resp = self.iot_data.get_thing_shadow(thingName=thing_name)
        return json.loads(resp["payload"].read())

    def update_shadow_desired(self, thing_name: str, state: Dict):
        """Push desired state to device shadow."""
        payload = json.dumps({"state": {"desired": state}})
        self.iot_data.update_thing_shadow(thingName=thing_name, payload=payload)
        print(f"[AWS IoT] Shadow updated for {thing_name}: {state}")

    def create_ota_job(
        self,
        job_id: str,
        target_thing_arns: List[str],
        firmware_s3_bucket: str,
        firmware_s3_key: str,
        rollout_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Create an AWS IoT OTA update job for a set of devices.
        Devices poll the job endpoint and download firmware from S3.
        """
        document = {
            "operation": "firmware_update",
            "firmwareLocation": {
                "bucket": firmware_s3_bucket,
                "key": firmware_s3_key,
            },
            "updateTimestamp": int(time.time()),
        }
        kwargs = {
            "jobId": job_id,
            "targets": target_thing_arns,
            "document": json.dumps(document),
            "description": f"OTA firmware update {job_id}",
            "targetSelection": "SNAPSHOT",
        }
        if rollout_config:
            kwargs["jobExecutionsRolloutConfig"] = rollout_config

        resp = self.iot.create_job(**kwargs)
        print(f"[AWS IoT] OTA job created: {resp['jobId']}")
        return resp

    def upload_firmware_to_s3(self, local_path: str, bucket: str, key: str) -> str:
        """Upload firmware binary to S3 for OTA distribution."""
        self.s3.upload_file(local_path, bucket, key)
        url = f"s3://{bucket}/{key}"
        print(f"[AWS IoT] Firmware uploaded: {url}")
        return url
