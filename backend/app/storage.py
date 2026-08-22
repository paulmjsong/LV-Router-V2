from __future__ import annotations

import asyncio
import re
from pathlib import Path

import boto3

from .config import Settings


class ObjectStore:
    async def initialize(self) -> None:
        """Verify that the object store is ready for use."""
        raise NotImplementedError

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        raise NotImplementedError


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    async def initialize(self) -> None:
        await asyncio.to_thread(
            self.root.mkdir,
            parents=True,
            exist_ok=True,
        )

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        del content_type

        target = (self.root / key).resolve()

        if self.root not in target.parents:
            raise ValueError("Unsafe object key")

        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

        return f"file://{target}"


class S3ObjectStore(ObjectStore):
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket

        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
        )

    async def initialize(self) -> None:
        # minio-init is responsible for creating the bucket.
        # Here we only verify that it exists and is accessible.
        await asyncio.to_thread(
            self.client.head_bucket,
            Bucket=self.bucket,
        )

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

        return f"s3://{self.bucket}/{key}"


def build_object_store(settings: Settings) -> ObjectStore:
    if settings.object_store == "s3":
        return S3ObjectStore(settings)

    return LocalObjectStore(settings.local_object_store_path)


def safe_filename(filename: str) -> str:
    base = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned[:180] or "document"