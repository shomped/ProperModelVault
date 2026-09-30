import hashlib
import json
import logging
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

@runtime_checkable
class ModelSerializer(Protocol):
    def save(self, model: Any, path: Path) -> None: ...
    def load(self, path: Path, model: Optional[Any] = None) -> Any: ...
    @property
    def extension(self) -> str: ...


class PickleSerializer:
    extension = ".pkl"
    def save(self, model: Any, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    def load(self, path: Path, model: Optional[Any] = None) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)


class SafeTensorsSerializer:
    extension = ".safetensors"
    def save(self, model: Any, path: Path) -> None:
        from safetensors.torch import save_file
        if not hasattr(model, 'state_dict'):
            raise ValueError("Model must have a state_dict method")
        save_file(model.state_dict(), str(path))

    def load(self, path: Path, model: Optional[Any] = None) -> Any:
        from safetensors.torch import load_file
        if model is None:
            raise ValueError("SafeTensors requires a model instance to load weights into")
        
        state_dict = load_file(str(path))
        model.load_state_dict(state_dict, strict=False)
        return model

class ModelVault:
    """
    file-system based model registry with integrity checks
    supports pluggable serialisers like pickle or safetenzors
    uses atomic writes to prevent data corruotion during saves
    """
    def __init__(self, base_dir: str, serializer: Optional[ModelSerializer] = None):
        self.base_path = Path(base_dir).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._serializer = serializer or PickleSerializer()

    def _model_dir(self, name: str) -> Path:
        return self.base_path / name

    def _metadata_path(self, name: str, version: int) -> Path:
        return self._model_dir(name) / f"v{version}_meta.json"

    def _model_path(self, name: str, version: int) -> Path:
        return self._model_dir(name) / f"v{version}_model{self._serializer.extension}"

    def _checksum_path(self, name: str, version: int) -> Path:
        return self._model_dir(name) / f"v{version}_sha256.txt"

    def _next_version(self, name: str) -> int:
        versions = self.list_versions(name)
        return max(versions) + 1 if versions else 1

    def save(self, name: str, model: Any, metadata: Optional[Dict] = None) -> int:
        """
        save a model with optional metadata and sha-256 checksum
        performs atomic write using temp files to ensure integriy
        """
        version = self._next_version(name)
        model_dir = self._model_dir(name)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self._model_path(name, version).exists():
            raise FileExistsError(f"Version {version} of '{name}' already exists.")

        meta = dict(metadata or {})
        meta.update({"version": version})
        
        with tempfile.TemporaryDirectory(dir=str(model_dir)) as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            tmp_model = tmp_path / f"model{self._serializer.extension}"
            self._serializer.save(model, tmp_model)
            
            sha = hashlib.sha256(tmp_model.read_bytes()).hexdigest()
            tmp_checksum = tmp_path / "sha256.txt"
            tmp_checksum.write_text(sha)
            
            tmp_meta = tmp_path / "meta.json"
            meta["sha256"] = sha
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            
            final_model = self._model_path(name, version)
            final_checksum = self._checksum_path(name, version)
            final_meta = self._metadata_path(name, version)
            
            shutil.move(str(tmp_model), str(final_model))
            shutil.move(str(tmp_checksum), str(final_checksum))
            shutil.move(str(tmp_meta), str(final_meta))

        logger.info("Saved %s v%d (sha256=%s)", name, version, sha[:12])
        return version

    def load(self, name: str, version: int, model: Optional[Any] = None) -> Any:
        """
        load a specific version of a model with checksum verification
        if using safetenzors, provide the model architecture via 'model' arg
        """
        model_path = self._model_path(name, version)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model {name} version {version} not found.")
        
        checksum_path = self._checksum_path(name, version)
        if checksum_path.is_file():
            expected = checksum_path.read_text().strip()
            actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual != expected:
                logger.error(
                        "Integrity check failed for %s v%d. Expected: %s, got: %s",
                        name,version,expected[:12],actual[:12]
                raise RuntimeError(f"Checksum mismatch for {name} v{version}")

        result = self._serializer.load(model_path, model=model)
        logger.info("Loaded %s v%d", name, version)
        return result

    def load_latest(self, name: str, model: Optional[Any] = None) -> Any:
        """load the most recent version of a model"""
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(f"No versions found for model {name}.")
        return self.load(name, max(versions), model=model)

    def list_versions(self, name: str) -> List[int]:
        """list all saved versions for a model, sorted ascending"""
        model_dir = self._model_dir(name)
        if not model_dir.is_dir():
            return []
        
        versions = []
        ext = self._serializer.extension
        suffix = f"_model{ext}"
        
        for file in model_dir.iterdir():
            if file.name.startswith("v") and file.name.endswith(suffix):
                try:
                    v_str = file.name[1:-len(suffix)]
                    v = int(v_str)
                    versions.append(v)
                except ValueError:
                    continue
        return sorted(versions)

    def get_metadata(self, name: str, version: int) -> Dict:
        """retrieve metadata for a specific model version"""
        meta_path = self._metadata_path(name, version)
        if not meta_path.is_file():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete_version(self, name: str, version: int) -> None:
        """remove a specific version of a model and its metadata"""
        model_path = self._model_path(name, version)
        meta_path = self._metadata_path(name, version)
        checksum_path = self._checksum_path(name, version)
        
        for p in (model_path, meta_path, checksum_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
                
        model_dir = self._model_dir(name)
        if not any(model_dir.iterdir()):
            model_dir.rmdir()