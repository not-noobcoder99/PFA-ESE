"""
Model service layer for heart disease prediction
Handles model loading, inference, and confidence scoring
"""

import numpy as np
from pathlib import Path
from joblib import load
import logging
from typing import Tuple, List, Dict
from src.config import settings, RISK_LABELS
from src.logger import logger as app_logger


class ModelService:
    """Encapsulates ML model operations"""
    
    def __init__(self, model_path: str = settings.MODEL_PATH):
        """Initialize model service"""
        self.model_path = Path(model_path)
        self.model = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}\n"
                    f"Expected RandomForest.joblib from Assignment 2"
                )
            
            self.model = load(self.model_path)
            self.model_loaded = True
            app_logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            app_logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def predict_single(
        self,
        features: List[float]
    ) -> Tuple[int, float]:
        """
        Predict for single patient
        
        Args:
            features: List of 13 feature values in correct order
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Initialize service first.")
        
        try:
            import pandas as pd
            
            # Convert to DataFrame with feature names (required by pipeline)
            X = pd.DataFrame([features], columns=settings.FEATURE_ORDER)
            
            # Get prediction
            prediction = self.model.predict(X)[0]
            
            # Get probability scores
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            
            return int(prediction), confidence
            
        except Exception as e:
            app_logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    
    def predict_batch(
        self,
        features_list: List[List[float]]
    ) -> Tuple[List[int], List[float]]:
        """
        Predict for batch of patients
        
        Args:
            features_list: List of feature lists
            
        Returns:
            Tuple of (predictions, confidences)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Initialize service first.")
        
        try:
            import pandas as pd
            
            # Convert to DataFrame with feature names
            X = pd.DataFrame(features_list, columns=settings.FEATURE_ORDER)
            
            predictions = self.model.predict(X)
            probas = self.model.predict_proba(X)
            confidences = np.max(probas, axis=1)
            
            return [int(p) for p in predictions], [float(c) for c in confidences]
            
        except Exception as e:
            app_logger.error(f"Batch prediction failed: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from RandomForest
        
        Returns:
            Dict mapping feature names to importance values
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Check if model has feature_importances_ (ensemble models)
            if hasattr(self.model, 'named_steps'):
                # Pipeline model
                rf_model = self.model.named_steps.get('random_forest')
                if rf_model and hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                else:
                    raise AttributeError("Model does not have feature importances")
            else:
                importances = self.model.feature_importances_
            
            feature_importance = {
                feature: float(imp)
                for feature, imp in zip(settings.FEATURE_ORDER, importances)
            }
            
            return dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )
        except Exception as e:
            app_logger.error(f"Could not get feature importance: {str(e)}")
            return {}
    
    def health_check(self) -> Dict[str, any]:
        """Verify model is functional"""
        if not self.model_loaded:
            return {"status": "offline", "error": "Model not loaded"}
        
        try:
            # Test with dummy data
            test_features = [[50, 1, 0, 130, 200, 0, 1, 120, 0, 1.0, 1, 0, 2]]
            pred, conf = self.predict_single(test_features[0])
            
            return {
                "status": "healthy",
                "model_type": type(self.model).__name__,
                "test_prediction": int(pred),
                "test_confidence": float(conf)
            }
        except Exception as e:
            return {"status": "degraded", "error": str(e)}


# Global model service instance
_model_service = None


def get_model_service() -> ModelService:
    """Get or create global model service instance"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
