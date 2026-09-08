# Author: Prof. Anis Koubaa <anis.koubaa@gmail.com>
"""
RAISE 2026 — VLA client factory: one switch picks the brain.

The Lab 2.2 executor calls make_vla_client(...) and gets back "something that
acts" — it never imports a concrete backend. Defaults make everything run FULLY
LOCAL with no server and no network: if VLA_BACKEND is unset you get in-process
SmolVLA. Set VLA_BACKEND=remote-* + VLA_REMOTE_URL to flip to a self-hosted
endpoint, and nothing else in the executor changes.

Config contract (see LOCAL_VS_REMOTE_VLA.md):
    VLA_BACKEND     local-smolvla (default) | remote-openvla | remote-pi0
    VLA_LOCAL_CKPT  lerobot/smolvla_base (default) | a fine-tuned output_dir
    VLA_REMOTE_URL  required only for remote-*  (e.g. http://gpu-box.lan:8000)
    VLA_DEVICE      cuda if available else cpu
"""

from __future__ import annotations

import os


def _default_device() -> str:
    """cuda when torch sees a GPU, else cpu — without importing torch eagerly."""
    if os.getenv('VLA_DEVICE'):
        return os.environ['VLA_DEVICE']
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def make_vla_client(backend: str | None = None):
    """Build the configured VLA backend.

    `backend` overrides VLA_BACKEND (which itself defaults to local-smolvla).
    Returns an object implementing the VLAClient protocol from base.py.
    """
    backend = (backend or os.getenv('VLA_BACKEND', 'local-smolvla')).lower()

    if backend in ('local-smolvla', 'local'):
        from .local_smolvla import LocalSmolVLA
        return LocalSmolVLA(
            ckpt=os.getenv('VLA_LOCAL_CKPT', 'lerobot/smolvla_base'),
            device=_default_device(),
        )

    if backend.startswith('remote'):
        # remote-openvla / remote-pi0 / remote-smolvla all speak the same HTTP
        # contract — the server decides which model it hosts.
        from .remote_vla import RemoteVLA
        return RemoteVLA(url=os.getenv('VLA_REMOTE_URL', ''))

    raise ValueError(
        f"Unknown VLA_BACKEND '{backend}'. "
        "Use local-smolvla (default) | remote-openvla | remote-pi0."
    )
