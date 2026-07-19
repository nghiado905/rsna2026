#!/usr/bin/env python3
"""Smoke-test Google Drive access with OAuth 2.0.

The script lists a target Drive folder, uploads a small text file, downloads it
again, and verifies that the downloaded bytes match the upload.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


SCOPES = ("https://www.googleapis.com/auth/drive",)


def build_drive_client(client_secrets_json: Path, open_browser: bool):
    creds = None
    # File token.pickle lưu trữ token truy cập sau lần đăng nhập đầu tiên
    token_path = client_secrets_json.parent / "token.pickle"

    if token_path.exists():
        with token_path.open("rb") as token:
            creds = pickle.load(token)

    # Nếu chưa có token hoặc token hết hạn, mở trình duyệt để đăng nhập
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets_json), list(SCOPES)
            )
            creds = flow.run_local_server(
                port=0,
                open_browser=open_browser,
                authorization_prompt_message=(
                    "Open this URL to authorize Google Drive access:\n{url}\n"
                ),
                timeout_seconds=300,
            )
        
        # Lưu lại token để các lần sau chạy không cần mở trình duyệt nữa
        with token_path.open("wb") as token:
            pickle.dump(creds, token)

    return build("drive", "v3", credentials=creds)


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
        "--client-secrets-json",
        default=os.environ.get("GOOGLE_CLIENT_SECRETS"),
        help="Path to the OAuth 2.0 client secrets JSON. Defaults to GOOGLE_CLIENT_SECRETS.",
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
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the OAuth URL instead of opening the browser automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.client_secrets_json:
        raise SystemExit("Missing --client-secrets-json or GOOGLE_CLIENT_SECRETS")
    if not args.folder_id:
        raise SystemExit("Missing --folder-id or GOOGLE_DRIVE_FOLDER_ID")

    client_secrets_json = Path(args.client_secrets_json).expanduser().resolve()
    if not client_secrets_json.is_file():
        raise FileNotFoundError(client_secrets_json)

    work_dir = Path(args.work_dir).expanduser().resolve()
    upload_path = work_dir / "google_drive_smoke_upload.txt"
    download_path = work_dir / "google_drive_smoke_download.txt"
    payload = b"rsna google drive smoke test via oauth\n"
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(payload)

    service = build_drive_client(client_secrets_json, open_browser=not args.no_browser)
    list_folder(service, args.folder_id, args.list_limit)

    try:
        uploaded_id = upload_file(
            service,
            args.folder_id,
            upload_path,
            "google_drive_smoke_upload.txt",
        )
    except HttpError as exc:
        raise RuntimeError(f"Upload failed: {exc}") from exc

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
