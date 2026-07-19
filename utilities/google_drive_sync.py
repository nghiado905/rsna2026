from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Dict

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = ("https://www.googleapis.com/auth/drive",)
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def _load_credentials(credentials_file: Path, token_file: Path):
    creds = None
    if token_file.is_file():
        with token_file.open("rb") as f:
            creds = pickle.load(f)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with token_file.open("wb") as f:
            pickle.dump(creds, f)

    if not creds or not creds.valid:
        raise RuntimeError(
            "Google Drive OAuth token is missing or invalid. Run "
            "tools/google_drive_smoke_test.py locally first to create token.pickle."
        )

    return creds


def build_drive_service(credentials_file: str | Path, token_file: str | Path):
    credentials_path = Path(credentials_file).expanduser().resolve()
    token_path = Path(token_file).expanduser().resolve()
    if not credentials_path.is_file():
        raise FileNotFoundError(credentials_path)
    if not token_path.is_file():
        # Validate that credentials are an OAuth client file. Login is intentionally
        # not started from training because Kaggle/non-interactive runs cannot finish it.
        InstalledAppFlow.from_client_secrets_file(str(credentials_path), list(SCOPES))
    creds = _load_credentials(credentials_path, token_path)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _escape_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _find_child(service, parent_id: str, name: str, mime_type: str | None = None) -> str | None:
    query = [
        f"'{_escape_query_value(parent_id)}' in parents",
        "trashed = false",
        f"name = '{_escape_query_value(name)}'",
    ]
    if mime_type is not None:
        query.append(f"mimeType = '{mime_type}'")

    response = (
        service.files()
        .list(
            q=" and ".join(query),
            fields="files(id, name)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = response.get("files", [])
    return files[0]["id"] if files else None


def _ensure_child_folder(service, parent_id: str, folder_name: str) -> str:
    existing_id = _find_child(service, parent_id, folder_name, FOLDER_MIME_TYPE)
    if existing_id is not None:
        return existing_id

    created = (
        service.files()
        .create(
            body={
                "name": folder_name,
                "mimeType": FOLDER_MIME_TYPE,
                "parents": [parent_id],
            },
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    return created["id"]


def _ensure_folder_path(service, root_folder_id: str, parts: tuple[str, ...]) -> str:
    folder_id = root_folder_id
    for part in parts:
        folder_id = _ensure_child_folder(service, folder_id, part)
    return folder_id


def _upload_or_update_file(service, parent_id: str, local_file: Path) -> str:
    existing_id = _find_child(service, parent_id, local_file.name)
    media = MediaFileUpload(str(local_file), resumable=True)

    if existing_id is not None:
        updated = (
            service.files()
            .update(
                fileId=existing_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        return updated["id"]

    created = (
        service.files()
        .create(
            body={"name": local_file.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    return created["id"]


def sync_directory_to_drive(
    local_dir: str | Path,
    drive_folder_id: str,
    credentials_file: str | Path,
    token_file: str | Path,
    remote_prefix_parts: tuple[str, ...] = (),
    log: Callable[..., None] = print,
) -> None:
    local_root = Path(local_dir).resolve()
    if not local_root.is_dir():
        raise NotADirectoryError(local_root)

    service = build_drive_service(credentials_file, token_file)
    folder_cache: Dict[tuple[str, ...], str] = {}
    uploaded = 0

    for local_file in sorted(p for p in local_root.rglob("*") if p.is_file()):
        relative_parent = local_file.parent.relative_to(local_root)
        folder_parts = remote_prefix_parts + tuple(relative_parent.parts)
        if folder_parts not in folder_cache:
            folder_cache[folder_parts] = _ensure_folder_path(
                service, drive_folder_id, folder_parts
            )
        _upload_or_update_file(service, folder_cache[folder_parts], local_file)
        uploaded += 1

    log(f"Google Drive sync completed: {uploaded} file(s) from {local_root}")
