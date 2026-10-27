"""
engine/checkpoint_manager.py
============================
Dual-Track Checkpoint & Multi-Tenant Fast-Weight LRU Session Manager.

Responsibilities:
1. Base Checkpoint (`checkpoints/base_uttt.pt`):
   Loads the shared Slow Weights (Theta_slow) and Meta-Initialized Fast Weights (W_0)
   once into RTX 5090 VRAM and freezes all Slow Weights.
2. Multi-User Isolated Fast-Weight Pool with 3-Tier LRU Offloading:
   - Tier 1 (GPU VRAM Active LRU): Up to `max_gpu_users` concurrent `FastWeightState`s.
   - Tier 2 (CPU RAM Warm LRU): Up to `max_cpu_users` offloaded `FastWeightState`s.
   - Tier 3 (Disk Cold Storage): `personas/{user_id}_state.pt` (and `personas/{user_id}.pt`).
3. Interactive Lifecycle Commands:
   Supports `/save`, `/reset`, `/snapshot`, and `/diff` (Frobenius drift from Base W_0).
"""

from __future__ import annotations

import os
import hashlib
import re
import shutil
import threading
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from model.uttt_layer import FastWeightState
from model.uttt_model import uTTTConfig, uTTTModel


def _sanitize_id(raw_id: str) -> str:
    """Reject ambiguous identifiers instead of merging distinct users silently."""
    cleaned = raw_id.strip()
    if not cleaned or cleaned in {".", ".."} or not re.fullmatch(r"[a-zA-Z0-9_.-]+", cleaned):
        raise ValueError("User IDs and snapshot tags must contain only letters, digits, _, -, or .")
    return cleaned


class DualTrackCheckpointManager:
    """
    Manages the single shared Base Model (Slow Weights + W_0 in GPU VRAM) and
    per-user isolated Fast-Weight states with automatic GPU -> CPU -> Disk LRU eviction.
    """

    def __init__(
        self,
        base_checkpoint_path: str | Path = "checkpoints/base_uttt.pt",
        personas_dir: str | Path = "personas",
        config_path: str | Path = "configs/5090_config.yaml",
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_gpu_users: int = 16,
        max_cpu_users: int = 128,
        auto_create_base_if_missing: bool = True,
        lightweight_fallback: bool = False,
    ) -> None:
        self.base_checkpoint_path = Path(base_checkpoint_path)
        self.personas_dir = Path(personas_dir)
        self.snapshots_dir = self.personas_dir / "snapshots"
        self.config_path = Path(config_path)

        self.personas_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if dtype is None:
            self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = dtype

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.max_gpu_users = max(1, int(max_gpu_users))
        self.max_cpu_users = max(1, int(max_cpu_users))

        # Thread-safe LRU caches mapping user_id -> FastWeightState
        self._lock = threading.RLock()
        self._disk_lock = threading.Lock()
        self._gpu_lru: OrderedDict[str, FastWeightState] = OrderedDict()
        self._cpu_lru: OrderedDict[str, FastWeightState] = OrderedDict()
        self._user_metadata: Dict[str, Dict[str, Any]] = {}

        # Load or initialize the shared Base Model
        self.model, self.config = self._load_or_init_base_model(
            auto_create=auto_create_base_if_missing,
            lightweight_fallback=lightweight_fallback,
        )
        with torch.no_grad():
            w0_sum = float(self.model.fast_weight_pool.w_up_init.detach().float().abs().sum().item())
        self.base_signature = f"{self.config.d_model}x{self.config.d_ff_fast}:{w0_sum:.6f}"
        # Retain the legacy signature for existing personas, but fingerprint all
        # checkpoint bytes for new saves (including slow and gate/down weights).
        if not hasattr(self, "base_fingerprint"):
            with self.base_checkpoint_path.open("rb") as source:
                self.base_fingerprint = self._fingerprint(source)

    @staticmethod
    def _fingerprint(source) -> str:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()

    # --------------------------------------------------------------------------
    # Base Checkpoint Loading & Initialization
    # --------------------------------------------------------------------------

    def _load_or_init_base_model(
        self, auto_create: bool = True, lightweight_fallback: bool = False
    ) -> Tuple[uTTTModel, uTTTConfig]:
        """
        Load `checkpoints/base_uttt.pt` into GPU memory and freeze Slow Weights.
        If missing and `auto_create=True`, initializes a fresh Base Checkpoint from config.
        """
        if self.base_checkpoint_path.exists():
            # Use the same open file for loading and hashing, even if training
            # atomically replaces the checkpoint path while startup is in progress.
            with self.base_checkpoint_path.open("rb") as source:
                ckpt = torch.load(source, map_location="cpu", weights_only=False)
                source.seek(0)
                self.base_fingerprint = self._fingerprint(source)
            cfg_dict = ckpt.get("config", {})
            config = uTTTConfig.from_dict(cfg_dict)
            model = uTTTModel(config)

            if "slow_weights" in ckpt and "init_fast_weights" in ckpt:
                model.load_dual_base_state(
                    slow_weights=ckpt["slow_weights"],
                    init_fast_weights=ckpt["init_fast_weights"],
                    strict=True,
                )
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
            else:
                raise ValueError(f"Unsupported base checkpoint format: {self.base_checkpoint_path}")

            model.to(device=self.device, dtype=self.dtype)
            model.freeze_slow_weights()
            model.eval()
            print(
                f"[CheckpointManager] Loaded Base Checkpoint from {self.base_checkpoint_path} "
                f"onto {self.device} ({self.dtype}) | Slow Weights Frozen."
            )
            return model, config

        if not auto_create:
            raise FileNotFoundError(
                f"Base checkpoint not found at {self.base_checkpoint_path}"
            )

        # Initialize from YAML config when cold-starting without an existing checkpoint
        if self.config_path.exists():
            config = uTTTConfig.from_yaml(self.config_path)
        else:
            config = uTTTConfig()

        if lightweight_fallback:
            config.d_model = 256
            config.n_layers = 4
            config.n_heads = 4
            config.n_kv_heads = 2
            config.d_ff_fast = 512
            config.chunk_size = 32
            config.sliding_window_size = 128

        model = uTTTModel(config).to(device=self.device, dtype=self.dtype)
        model.freeze_slow_weights()
        model.eval()

        # Persist initial Base Checkpoint so dual-track invariant holds immediately
        self.base_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_type": "uTTT_Base_Checkpoint",
            "version": "1.0.0",
            "step": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": config.to_dict(),
            "slow_weights": model.slow_state_dict(),
            "init_fast_weights": model.init_fast_state_dict(),
            "metrics": {"cold_initialized": True},
        }
        torch.save(payload, self.base_checkpoint_path)
        print(
            f"[CheckpointManager] Initialized & saved new Base Checkpoint at "
            f"{self.base_checkpoint_path}"
        )
        return model, config

    # --------------------------------------------------------------------------
    # Persona File Path Resolution (`{user_id}_state.pt` & `{user_id}.pt`)
    # --------------------------------------------------------------------------

    def get_persona_paths(self, user_id: str) -> Tuple[Path, Path]:
        """
        Return both canonical path (`personas/{user_id}_state.pt`) and alias path
        (`personas/{user_id}.pt`) for full compatibility across specifications.
        """
        safe_uid = _sanitize_id(user_id)
        primary = self.personas_dir / f"{safe_uid}_state.pt"
        alias = self.personas_dir / f"{safe_uid}.pt"
        return primary, alias

    def _find_existing_persona_file(self, user_id: str) -> Optional[Path]:
        primary, alias = self.get_persona_paths(user_id)
        if primary.exists():
            return primary
        if not user_id.endswith("_state") and alias.exists():
            return alias
        return None

    # --------------------------------------------------------------------------
    # Multi-User Session Lifecycle & 3-Tier LRU Cache Management
    # --------------------------------------------------------------------------

    def _evict_gpu_lru_if_needed(self, exclude_user: Optional[str] = None) -> None:
        """
        Evict least-recently-used user FastWeightState from GPU VRAM to CPU RAM & Disk
        when active GPU sessions exceed `max_gpu_users`.
        """
        while len(self._gpu_lru) >= self.max_gpu_users:
            victim_uid, victim_state = next(iter(self._gpu_lru.items()))
            if victim_uid == exclude_user and len(self._gpu_lru) == 1:
                break
            # Persist to disk before offloading to CPU RAM
            self._save_state_to_disk(victim_uid, victim_state)
            self._gpu_lru.pop(victim_uid)

            cpu_state = victim_state.to(device="cpu", dtype=self.dtype, non_blocking=False)
            self._cpu_lru[victim_uid] = cpu_state
            self._cpu_lru.move_to_end(victim_uid)

            while len(self._cpu_lru) > self.max_cpu_users:
                self._cpu_lru.popitem(last=False)

    def get_user_state(self, user_id: str) -> FastWeightState:
        """
        Retrieve a user's `FastWeightState` on the active compute device (GPU VRAM):
        1. Hit in GPU LRU -> move to MRU and return.
        2. Hit in CPU LRU -> promote to GPU LRU (evicting oldest GPU user if full).
        3. Hit on Disk (`personas/{user_id}_state.pt` or `.pt`) -> load to GPU LRU.
        4. Cold Start -> clone W_0 from `model.fast_weight_pool` into GPU LRU.
        """
        safe_uid = _sanitize_id(user_id)
        with self._lock:
            # Tier 1: GPU VRAM Hit
            if safe_uid in self._gpu_lru:
                self._gpu_lru.move_to_end(safe_uid)
                return self._gpu_lru[safe_uid]

            # Prepare slot in GPU VRAM
            self._evict_gpu_lru_if_needed(exclude_user=safe_uid)

            # Tier 2: CPU RAM Hit
            if safe_uid in self._cpu_lru:
                cpu_state = self._cpu_lru.pop(safe_uid)
                gpu_state = cpu_state.to(device=self.device, dtype=self.dtype)
                self._gpu_lru[safe_uid] = gpu_state
                return gpu_state

            # Tier 3: Disk Persona Checkpoint Hit
            disk_path = self._find_existing_persona_file(safe_uid)
            if disk_path is not None:
                try:
                    payload = torch.load(disk_path, map_location="cpu", weights_only=False)
                    if payload.get("user_id", safe_uid) != safe_uid:
                        raise ValueError("Persona file belongs to another user")
                    saved_sig = payload.get("base_signature")
                    saved_fingerprint = payload.get("base_fingerprint")
                    compatible = (
                        saved_fingerprint == self.base_fingerprint
                        if saved_fingerprint is not None
                        else saved_sig == self.base_signature
                    )
                    if compatible:
                        fast_dict = payload.get("fast_state", payload)
                        gpu_state = FastWeightState.from_state_dict(
                            fast_dict, device=self.device, dtype=self.dtype
                        )
                        if (
                            self._valid_state_shapes(gpu_state)
                        ):
                            self._user_metadata[safe_uid] = payload.get("metadata", {})
                            self._gpu_lru[safe_uid] = gpu_state
                            return gpu_state
                    print(
                        f"[CheckpointManager] Persona '{safe_uid}' on disk was created under a "
                        "different or incompatible Base Checkpoint; "
                        "auto-resetting to current Base W_0.",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[CheckpointManager] Could not load persona '{safe_uid}' from {disk_path} ({exc}); "
                        "re-initializing from Base W_0.",
                        flush=True,
                    )

            # Cold Start: Clone W_0 from SharedFastWeightPool
            fresh_state = self.model.init_fast_state(
                batch_size=1,
                device=self.device,
                dtype=self.dtype,
                detach=True,
            )
            self._user_metadata[safe_uid] = {
                "user_id": safe_uid,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "turns": 0,
            }
            self._gpu_lru[safe_uid] = fresh_state
            return fresh_state

    def update_user_state(
        self,
        user_id: str,
        new_state: FastWeightState,
        auto_save: bool = True,
    ) -> Optional[Path]:
        """Update in-memory GPU LRU state for `user_id` and optionally persist to disk."""
        safe_uid = _sanitize_id(user_id)
        with self._lock:
            gpu_state = new_state.detach().to(device=self.device, dtype=self.dtype)
            if safe_uid not in self._gpu_lru:
                self._evict_gpu_lru_if_needed(exclude_user=safe_uid)
            self._gpu_lru[safe_uid] = gpu_state
            self._gpu_lru.move_to_end(safe_uid)
            self._cpu_lru.pop(safe_uid, None)

            meta = self._user_metadata.setdefault(
                safe_uid,
                {
                    "user_id": safe_uid,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "turns": 0,
                },
            )
            meta["turns"] = int(meta.get("turns", 0)) + 1
            meta.pop("_cached_drift", None)
            meta.pop("_cached_drift_step", None)
            meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            if auto_save:
                # A successful response means the latest state is durable. Keeping
                # writes ordered also prevents an older turn overwriting /reset.
                return self._save_state_to_disk(safe_uid, gpu_state)
            return None

    def _write_payload_files(self, primary_path: Path, alias_path: Path, payload: Dict[str, Any]) -> None:
        """Thread-safe atomic disk writer with fsync and hardlink alias optimization."""
        with self._disk_lock:
            tmp_path = Path(str(primary_path) + ".tmp")
            with open(tmp_path, "wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, primary_path)

            # alice_state.pt is alice's canonical file, not an alias we may
            # overwrite when saving a different user named alice_state.
            if payload["user_id"].endswith("_state"):
                return
            # Synchronize `{user_id}.pt` via zero-copy hardlink (fallback to copy2 if cross-device)
            try:
                if alias_path.exists() or alias_path.is_symlink():
                    alias_path.unlink()
                os.link(primary_path, alias_path)
            except Exception:
                try:
                    shutil.copy2(primary_path, alias_path)
                except Exception:
                    pass

    def _save_state_to_disk(
        self, user_id: str, state: FastWeightState
    ) -> Path:
        """Internal atomic writer for `personas/{user_id}_state.pt` and `{user_id}.pt`."""
        safe_uid = _sanitize_id(user_id)
        primary_path, alias_path = self.get_persona_paths(safe_uid)
        drift_info = self.model.fast_weight_pool.compute_drift(state)

        meta = self._user_metadata.setdefault(safe_uid, {"user_id": safe_uid})
        meta["_cached_drift"] = dict(drift_info)
        meta["_cached_drift_step"] = state.step

        payload = {
            "checkpoint_type": "uTTT_User_Persona_State",
            "version": "1.0.0",
            "user_id": safe_uid,
            "base_signature": self.base_signature,
            "base_fingerprint": self.base_fingerprint,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fast_state": state.to_state_dict(),
            "drift_metrics": drift_info,
            "metadata": {k: v for k, v in meta.items() if not k.startswith("_")},
        }

        self._write_payload_files(primary_path, alias_path, payload)

        return primary_path

    # --------------------------------------------------------------------------
    # Interactive Control Commands (/save, /reset, /snapshot, /diff)
    # --------------------------------------------------------------------------

    def save_user_state(self, user_id: str) -> Path:
        """Persist current user's FastWeightState to `personas/{user_id}_state.pt` (`/save`)."""
        with self._lock:
            state = self.get_user_state(user_id)
            return self._save_state_to_disk(user_id, state)

    def reset_user_state(self, user_id: str, persist: bool = True) -> FastWeightState:
        """
        Reset user's FastWeightState and Momentum buffers back to Base Checkpoint W_0 (`/reset`).
        """
        safe_uid = _sanitize_id(user_id)
        with self._lock:
            if safe_uid not in self._gpu_lru:
                self._evict_gpu_lru_if_needed(exclude_user=safe_uid)
            fresh_state = self.model.init_fast_state(
                batch_size=1,
                device=self.device,
                dtype=self.dtype,
                detach=True,
            )
            self._gpu_lru[safe_uid] = fresh_state
            self._gpu_lru.move_to_end(safe_uid)
            self._cpu_lru.pop(safe_uid, None)
            self._user_metadata[safe_uid] = {
                "user_id": safe_uid,
                "reset_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "turns": 0,
            }
            if persist:
                self._save_state_to_disk(safe_uid, fresh_state)
            return fresh_state

    def save_snapshot(self, user_id: str, snapshot_tag: Optional[str] = None) -> Path:
        """
        Save an immutable stage snapshot of the user's current persona fast weights (`/snapshot`).
        """
        safe_uid = _sanitize_id(user_id)
        tag = _sanitize_id(snapshot_tag) if snapshot_tag else time.strftime("%Y%m%d_%H%M%S")
        with self._lock:
            state = self.get_user_state(safe_uid)
            drift_info = self.model.fast_weight_pool.compute_drift(state)
            user_snapshots = self.snapshots_dir / safe_uid
            user_snapshots.mkdir(parents=True, exist_ok=True)
            snap_path = user_snapshots / f"{tag}.pt"
            payload = {
                "checkpoint_type": "uTTT_Persona_Snapshot",
                "user_id": safe_uid,
                "snapshot_tag": tag,
                "base_signature": self.base_signature,
                "base_fingerprint": self.base_fingerprint,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fast_state": state.to_state_dict(),
                "drift_metrics": drift_info,
            }
            # Publish a complete snapshot atomically without replacing a tag.
            with tempfile.NamedTemporaryFile(dir=user_snapshots, suffix=".tmp", delete=False) as destination:
                temporary_path = Path(destination.name)
                try:
                    torch.save(payload, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                except BaseException:
                    destination.close()
                    temporary_path.unlink(missing_ok=True)
                    raise
            try:
                os.link(temporary_path, snap_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            return snap_path

    def load_snapshot(self, user_id: str, snapshot_tag: str) -> FastWeightState:
        """Restore a previously saved persona snapshot into the active GPU session."""
        safe_uid = _sanitize_id(user_id)
        tag = _sanitize_id(snapshot_tag)
        snap_path = self.snapshots_dir / safe_uid / f"{tag}.pt"
        if not snap_path.exists():
            snap_path = self.snapshots_dir / f"{safe_uid}_{tag}.pt"
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_path}")

        with self._lock:
            payload = torch.load(snap_path, map_location="cpu", weights_only=False)
            if payload.get("user_id") != safe_uid or payload.get("snapshot_tag") != tag:
                raise ValueError("Snapshot belongs to a different user or tag")
            if payload.get("base_fingerprint") != self.base_fingerprint:
                raise ValueError("Snapshot base checkpoint is different or cannot be verified")
            restored = FastWeightState.from_state_dict(
                payload["fast_state"], device=self.device, dtype=self.dtype
            )
            if not self._valid_state_shapes(restored):
                raise ValueError("Snapshot fast-weight shapes do not match the base model")
            self.update_user_state(safe_uid, restored, auto_save=True)
            return restored

    def _valid_state_shapes(self, state: FastWeightState) -> bool:
        d, f = self.config.d_model, self.config.d_ff_fast
        expected = {
            "w_up": (1, d, f), "w_gate": (1, d, f), "w_down": (1, f, d),
            "s_up": (1, d, f), "s_gate": (1, d, f), "s_down": (1, f, d),
            "ln_weight": (d,), "ln_bias": (d,),
        }
        return all(tuple(getattr(state, name).shape) == shape for name, shape in expected.items())

    def compute_user_drift(self, user_id: str) -> Dict[str, Any]:
        """
        Compute Frobenius norm drift ||W_user - W_0||_F and relative percentage (`/diff`).
        """
        safe_uid = _sanitize_id(user_id)
        with self._lock:
            state = self.get_user_state(safe_uid)
            meta = self._user_metadata.get(safe_uid, {})
            if meta.get("_cached_drift_step") == state.step and "_cached_drift" in meta:
                drift = dict(meta["_cached_drift"])
                drift["last_surprise_loss"] = state.last_surprise_loss
            else:
                drift = self.model.fast_weight_pool.compute_drift(state)
                meta["_cached_drift"] = dict(drift)
                meta["_cached_drift_step"] = state.step
            primary_path, _ = self.get_persona_paths(safe_uid)
            drift["user_id"] = safe_uid
            drift["persona_file"] = str(primary_path)
            drift["file_exists"] = primary_path.exists()
            return drift

    def pool_status(self) -> Dict[str, Any]:
        """Return diagnostic metrics for active GPU/CPU persona pools."""
        with self._lock:
            return {
                "device": str(self.device),
                "dtype": str(self.dtype),
                "gpu_active_users": list(self._gpu_lru.keys()),
                "gpu_pool_size": len(self._gpu_lru),
                "max_gpu_users": self.max_gpu_users,
                "cpu_cached_users": list(self._cpu_lru.keys()),
                "cpu_pool_size": len(self._cpu_lru),
                "max_cpu_users": self.max_cpu_users,
                "base_checkpoint": str(self.base_checkpoint_path),
            }
