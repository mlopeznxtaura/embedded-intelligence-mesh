"""
TimescaleDB time-series storage for sensor telemetry.
Continuous aggregates, retention policies, and Grafana dashboard provisioning.
SDKs: psycopg2, TimescaleDB, Redis
"""
import os
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import psycopg2
import psycopg2.extras
import pandas as pd


@dataclass
class SensorMetric:
    device: str
    timestamp: float
    label: str
    confidence: float
    rms: float
    temp_c: Optional[float] = None
    battery_pct: Optional[float] = None


class TimescaleStore:
    """
    High-throughput sensor data storage using TimescaleDB.
    Automatic partitioning, continuous aggregates, and data retention.
    """

    def __init__(self, conn_str: Optional[str] = None):
        self.conn_str = conn_str or os.environ.get(
            "TIMESCALE_URL",
            "postgresql://postgres:password@localhost:5432/sensors"
        )
        self.conn = psycopg2.connect(self.conn_str)
        self._setup_schema()
        print("[TimescaleDB] Connected and schema ready")

    def _setup_schema(self):
        with self.conn.cursor() as cur:
            # Main readings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    time        TIMESTAMPTZ NOT NULL,
                    device      TEXT NOT NULL,
                    label       TEXT,
                    confidence  DOUBLE PRECISION,
                    rms         DOUBLE PRECISION,
                    temp_c      DOUBLE PRECISION,
                    battery_pct DOUBLE PRECISION
                )
            """)
            # Convert to hypertable (TimescaleDB)
            cur.execute("""
                SELECT create_hypertable(
                    'sensor_readings', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                )
            """)
            # Index on device for fast per-device queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_device
                ON sensor_readings (device, time DESC)
            """)
            # Continuous aggregate: 1-minute rollup per device
            cur.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_1min
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('1 minute', time) AS bucket,
                    device,
                    AVG(rms)         AS avg_rms,
                    MAX(rms)         AS max_rms,
                    COUNT(*)         AS sample_count,
                    SUM(CASE WHEN label = 'anomaly' THEN 1 ELSE 0 END) AS anomaly_count
                FROM sensor_readings
                GROUP BY bucket, device
                WITH NO DATA
            """)
            self.conn.commit()

    def insert(self, metric: SensorMetric):
        """Insert a single sensor reading."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sensor_readings
                (time, device, label, confidence, rms, temp_c, battery_pct)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                datetime.fromtimestamp(metric.timestamp, tz=timezone.utc),
                metric.device, metric.label, metric.confidence,
                metric.rms, metric.temp_c, metric.battery_pct,
            ))
        self.conn.commit()

    def insert_batch(self, metrics: List[SensorMetric]):
        """Bulk insert for high-throughput ingestion."""
        rows = [
            (datetime.fromtimestamp(m.timestamp, tz=timezone.utc),
             m.device, m.label, m.confidence, m.rms, m.temp_c, m.battery_pct)
            for m in metrics
        ]
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO sensor_readings
                   (time, device, label, confidence, rms, temp_c, battery_pct)
                   VALUES %s""",
                rows,
            )
        self.conn.commit()
        print(f"[TimescaleDB] Inserted {len(metrics)} readings")

    def get_recent(self, device: str, minutes: int = 10) -> pd.DataFrame:
        """Fetch recent readings for a device."""
        since = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT time, label, confidence, rms, temp_c, battery_pct
                FROM sensor_readings
                WHERE device = %s AND time > %s
                ORDER BY time DESC
                LIMIT 1000
            """, (device, since))
            rows = cur.fetchall()
        return pd.DataFrame(rows)

    def get_anomaly_rate(self, device: str, hours: int = 24) -> float:
        """Anomaly rate for a device over last N hours."""
        since = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN label = 'anomaly' THEN 1 ELSE 0 END) as anomalies
                FROM sensor_readings
                WHERE device = %s AND time > %s
            """, (device, since))
            row = cur.fetchone()
        if row and row[0] > 0:
            return row[1] / row[0]
        return 0.0

    def fleet_summary(self) -> pd.DataFrame:
        """Summary stats for all active devices in last hour."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    device,
                    COUNT(*) as readings,
                    AVG(rms) as avg_rms,
                    MAX(rms) as max_rms,
                    SUM(CASE WHEN label = 'anomaly' THEN 1 ELSE 0 END) as anomalies,
                    MAX(time) as last_seen
                FROM sensor_readings
                WHERE time > NOW() - INTERVAL '1 hour'
                GROUP BY device
                ORDER BY anomalies DESC
            """)
            rows = cur.fetchall()
        return pd.DataFrame(rows)

    def set_retention_policy(self, days: int = 30):
        """Drop data older than N days (TimescaleDB retention policy)."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT add_retention_policy('sensor_readings', INTERVAL %s, if_not_exists => TRUE)",
                (f"{days} days",)
            )
        self.conn.commit()
        print(f"[TimescaleDB] Retention policy set: {days} days")

    def close(self):
        self.conn.close()
