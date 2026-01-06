"""
Model registry for versioning and lifecycle management.

Provides:
1. Model versioning
2. Metadata tracking
3. Model promotion (staging -> production)
4. Model retrieval
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..utils import setup_logging, ensure_dir, save_json, load_json

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
        self,
        model_path: str,
        model_version: str,
        metadata: Dict[str, Any],
        status: str = "staging"
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
        registry = load_json(str(self.registry_file))
        registry["models"].append(entry)
        save_json(registry, str(self.registry_file))
        
        self.logger.info(f"Model {model_version} registered with status={status}")
        
        return entry
    
    def promote_to_production(self, model_version: str) -> None:
        """
        Promote a model to production status.
        
        Args:
            model_version: Model version to promote
        """
        self.logger.info(f"Promoting {model_version} to production")
        
        registry = load_json(str(self.registry_file))
        
        # Demote current production model
        for model in registry["models"]:
            if model["status"] == "production":
                model["status"] = "archived"
                self.logger.info(f"  Archived previous production model: {model['model_version']}")
        
        # Promote new model
        for model in registry["models"]:
            if model["model_version"] == model_version:
                model["status"] = "production"
                model["promoted_at"] = datetime.now().isoformat()
                self.logger.info(f"  {model_version} is now in production")
                break
        
        save_json(registry, str(self.registry_file))
    
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
