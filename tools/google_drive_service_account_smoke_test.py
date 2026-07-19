#!/usr/bin/env python3
"""Smoke-test Google Drive access with a service account JSON key."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


SCOPES = ("https://www.googleapis.com/auth/drive",)


def build_drive_client(service_account_json: Path):
    credentials = service_account.Credentials.from_service_account_file(
        str(service_account_json),
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=credentials)


def list_folder(service, folder_id: str, limit: int) -> None:
    response = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name, mimeType, size, modifiedTime)",
            pageSize=limit,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )

    files = response.get("files", [])
    print(f"Found {len(files)} item(s) in folder {folder_id}:")
    for item in files:
        size = item.get("size", "-")
        print(f"- {item['name']} | id={item['id']} | size={size}")


def upload_file(service, folder_id: str, local_file: Path, drive_name: str) -> str:
    metadata = {"name": drive_name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_file), resumable=False)
    created = (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id, name",
            supportsAllDrives=True,
        )
        .execute()
    )
    print(f"Uploaded {local_file} as {created['name']} | id={created['id']}")
    return created["id"]


def download_file(service, file_id: str, output_file: Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download progress: {int(status.progress() * 100)}%")

    print(f"Downloaded file to {output_file}")


def delete_file(service, file_id: str) -> None:
    service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
    print(f"Deleted uploaded test file id={file_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-account-json",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to service account JSON. Defaults to GOOGLE_APPLICATION_CREDENTIALS.",
    )
    parser.add_argument(
        "--folder-id",
        default=os.environ.get("GOOGLE_DRIVE_FOLDER_ID"),
        help="Target Google Drive folder id. Defaults to GOOGLE_DRIVE_FOLDER_ID.",
    )
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("GOOGLE_DRIVE_TEST_WORK_DIR", "drive_test_tmp"),
        help="Local directory for temporary upload/download files.",
    )
    parser.add_argument("--list-limit", type=int, default=10)
    parser.add_argument(
        "--keep-uploaded",
        action="store_true",
        help="Keep the uploaded test file on Drive instead of deleting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.service_account_json:
        raise SystemExit("Missing --service-account-json or GOOGLE_APPLICATION_CREDENTIALS")
    if not args.folder_id:
        raise SystemExit("Missing --folder-id or GOOGLE_DRIVE_FOLDER_ID")

    service_account_json = Path(args.service_account_json).expanduser().resolve()
    if not service_account_json.is_file():
        raise FileNotFoundError(service_account_json)

    work_dir = Path(args.work_dir).expanduser().resolve()
    upload_path = work_dir / "google_drive_service_account_smoke_upload.txt"
    download_path = work_dir / "google_drive_service_account_smoke_download.txt"
    payload = b"rsna google drive smoke test via service account\n"
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(payload)

    service = build_drive_client(service_account_json)
    list_folder(service, args.folder_id, args.list_limit)

    try:
        uploaded_id = upload_file(
            service,
            args.folder_id,
            upload_path,
            "google_drive_service_account_smoke_upload.txt",
        )
    except HttpError as exc:
        if exc.resp.status == 403 and b"Service Accounts do not have storage quota" in exc.content:
            raise SystemExit(
                "Upload failed: this service account can read the folder, but Google "
                "Drive does not give service accounts storage quota in normal My Drive "
                "folders. Use a Shared Drive, or authenticate as a real user with OAuth."
            ) from exc
        raise

    try:
        download_file(service, uploaded_id, download_path)
        if download_path.read_bytes() != payload:
            raise RuntimeError("Downloaded content does not match uploaded content")
        print("Round-trip verification passed.")
    finally:
        if not args.keep_uploaded:
            delete_file(service, uploaded_id)


if __name__ == "__main__":
    main()
