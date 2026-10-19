"""
Model registry for versioning and lifecycle management.

Provides:
1. Model versioning
2. Metadata tracking
3. Model promotion (staging -> production)
4. Model retrieval
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import ensure_dir, load_json, save_json, setup_logging

logger = setup_logging(__name__)


class ModelRegistry:
    """
    Simple but production-ready model registry.

    Stores:
    - Model artifacts (pickled models)
    - Metadata (metrics, training config, feature schema)
    - Status (staging, production, archived)
    """

    def __init__(self, registry_path: str = "models"):
        """
        Initialize model registry.

        Args:
            registry_path: Base path for model storage
        """
        self.registry_path = Path(registry_path)
        self.registry_file = self.registry_path / "registry.json"
        ensure_dir(str(self.registry_path))

        # Initialize registry if doesn't exist
        if not self.registry_file.exists():
            save_json({"models": []}, str(self.registry_file))

        self.logger = logger

    def register_model(
        self, model_path: str, model_version: str, metadata: Dict[str, Any], status: str = "staging"
    ) -> Dict[str, Any]:
        """
        Register a new model in the registry.

        Args:
            model_path: Path to pickled model file
            model_version: Model version (e.g., "model_v1")
            metadata: Model metadata (metrics, config, etc.)
            status: Model status ("staging", "production", "archived")

        Returns:
            Registry entry for the model
        """
        self.logger.info(f"Registering model: {model_version}")

        registry = load_json(str(self.registry_file))
        if any(model["model_version"] == model_version for model in registry["models"]):
            raise ValueError(f"Model version already registered: {model_version}")

        if status not in {"staging", "production", "archived"}:
            raise ValueError(f"Invalid model status: {status}")
        # Create model version directory
        model_dir = self.registry_path / model_version
        ensure_dir(str(model_dir))

        # Copy model file only if source is different from destination
        model_dest = model_dir / "model.pkl"
        model_path_obj = Path(model_path)

        if model_path_obj.resolve() != model_dest.resolve():
            shutil.copy(model_path, model_dest)

        # Create registry entry
        entry = {
            "model_version": model_version,
            "registered_at": datetime.now().isoformat(),
            "status": status,
            "model_path": str(model_dest),
            "metadata": metadata,
        }

        # Save metadata
        metadata_file = model_dir / "metadata.json"
        save_json(entry, str(metadata_file))

        # Update registry
        registry["models"].append(entry)
        save_json(registry, str(self.registry_file))

        self.logger.info(f"Model {model_version} registered with status={status}")

        return entry

    def promote_to_production(
        self,
        model_version: str,
        reason: str = "manual promotion",
    ) -> Dict[str, Any]:
        """
        Promote a model to production status.

        Args:
            model_version: Model version to promote
        """
        self.logger.info(f"Promoting {model_version} to production")

        registry = load_json(str(self.registry_file))

        candidate = next(
            (model for model in registry["models"] if model["model_version"] == model_version),
            None,
        )
        if candidate is None:
            raise ValueError(f"Model version not found: {model_version}")

        previous_production = next(
            (model for model in registry["models"] if model["status"] == "production"),
            None,
        )

        # Demote current production model only after the candidate is validated.
        for model in registry["models"]:
            if model["status"] == "production" and model["model_version"] != model_version:
                model["status"] = "archived"
                self.logger.info(f"  Archived previous production model: {model['model_version']}")

        # Promote new model
        for model in registry["models"]:
            if model["model_version"] == model_version:
                model["status"] = "production"
                model["promoted_at"] = datetime.now().isoformat()
                model["promotion_reason"] = reason
                self.logger.info(f"  {model_version} is now in production")
                break

        save_json(registry, str(self.registry_file))

        return {
            "model_version": model_version,
            "previous_production_version": (
                previous_production["model_version"] if previous_production else None
            ),
            "promoted_at": candidate.get("promoted_at", datetime.now().isoformat()),
            "reason": reason,
        }

    def rollback_to_model(self, model_version: str, reason: str) -> Dict[str, Any]:
        """Restore a previously registered model as production with an audit reason."""
        if not reason.strip():
            raise ValueError("A rollback reason is required")
        return self.promote_to_production(model_version, reason=f"rollback: {reason}")

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the current production model.

        Returns:
            Production model entry, or None if no production model
        """
        registry = load_json(str(self.registry_file))

        for model in registry["models"]:
            if model["status"] == "production":
                return model

        self.logger.warning("No production model found")
        return None

    def get_model(self, model_version: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific model by version.

        Args:
            model_version: Model version

        Returns:
            Model entry, or None if not found
        """
        registry = load_json(str(self.registry_file))

        for model in registry["models"]:
            if model["model_version"] == model_version:
                return model

        return None

    def list_models(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all models, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of model entries
        """
        registry = load_json(str(self.registry_file))
        models = registry["models"]

        if status:
            models = [m for m in models if m["status"] == status]

        return models
