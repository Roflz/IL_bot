"""
Normalization Service

Handles normalized data views and fallback computation.
"""

import numpy as np
from typing import Optional, Tuple
from shared_pipeline.normalize import normalize_features


class NormalizationService:
    """Service for handling normalized data views."""
    
    def __init__(self, feature_mappings_file: str):
        """
        Initialize the normalization service.
        
        Args:
            feature_mappings_file: Path to feature mappings file
        """
        self.feature_mappings_file = feature_mappings_file
        self._fallback_normalized = None
        self._fallback_computed = False
    
    def get_normalized_sequences(self, 
                                raw_sequences: np.ndarray,
                                precomputed_normalized: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get normalized sequences, using precomputed if available or computing fallback.
        
        Args:
            raw_sequences: Raw input sequences (N, 10, 128)
            precomputed_normalized: Precomputed normalized sequences if available
            
        Returns:
            Normalized sequences (N, 10, 128)
        """
        if precomputed_normalized is not None:
            return precomputed_normalized
        
        # Compute fallback normalization
        if not self._fallback_computed:
            self._compute_fallback_normalization(raw_sequences)
        
        return self._fallback_normalized
    
    def _compute_fallback_normalization(self, raw_sequences: np.ndarray):
        """Compute fallback normalization for transient view."""
        print("⚠ Computing transient normalized view (fallback)")
        print("💡 This is computed in-memory and not saved")
        
        # Reshape to 2D for normalization
        original_shape = raw_sequences.shape
        sequences_2d = raw_sequences.reshape(-1, raw_sequences.shape[-1])
        
        # Apply normalization
        normalized_2d = normalize_features(sequences_2d, self.feature_mappings_file)
        
        # Reshape back to 3D
        self._fallback_normalized = normalized_2d.reshape(original_shape)
        self._fallback_computed = True
        
        print(f"✓ Transient normalized view computed: {self._fallback_normalized.shape}")
    
    def get_normalized_features(self, 
                               raw_features: np.ndarray,
                               precomputed_normalized: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get normalized features, using precomputed if available or computing fallback.
        
        Args:
            raw_features: Raw features (N, 128)
            precomputed_normalized: Precomputed normalized features if available
            
        Returns:
            Normalized features (N, 128)
        """
        if precomputed_normalized is not None:
            return precomputed_normalized
        
        # Compute fallback normalization
        if not self._fallback_computed:
            self._compute_fallback_normalization(raw_features.reshape(-1, 10, 128))
            # Extract the features from the computed sequences
            return self._fallback_normalized.reshape(-1, 128)
        
        return self._fallback_normalized.reshape(-1, 128)
    
    def is_using_fallback(self, precomputed_normalized: Optional[np.ndarray] = None) -> bool:
        """
        Check if we're using fallback normalization.
        
        Args:
            precomputed_normalized: Precomputed normalized data if available
            
        Returns:
            True if using fallback, False if using precomputed
        """
        return precomputed_normalized is None and self._fallback_computed
    
    def clear_fallback_cache(self):
        """Clear the fallback normalization cache."""
        self._fallback_normalized = None
        self._fallback_computed = False
