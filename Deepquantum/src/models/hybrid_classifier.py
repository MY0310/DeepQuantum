"""
Hybrid Quantum-Classical Classifier for Q-GAD system.

This module implements the fusion of quantum features with classical features
using ensemble methods (XGBoost/LightGBM) and neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import joblib


@dataclass
class HybridConfig:
    """Configuration for hybrid classifier."""
    quantum_feature_dim: int = 9  # Number of quantum features
    classical_feature_dim: int = 10  # Number of classical features
    hidden_dims: List[int] = None  # Hidden layer dimensions for NN branch
    use_xgboost: bool = True  # Use XGBoost as final classifier
    dropout: float = 0.1  # Dropout rate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class ClassicalFeatureEncoder(nn.Module):
    """
    Neural network encoder for classical tabular features.

    Processes traditional financial features:
    - Transaction amounts
    - Account age
    - Geographic risk scores
    - Temporal patterns
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical features.

        Args:
            x: Classical features [batch, input_dim]

        Returns:
            Encoded features [batch, output_dim]
        """
        return self.encoder(x)


class QuantumFeatureEncoder(nn.Module):
    """
    Neural network encoder for quantum features.

    Processes quantum statistics from GBS:
    - Photon number distributions
    - Entropy measures
    - Hafnian approximations
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode quantum features.

        Args:
            x: Quantum features [batch, input_dim]

        Returns:
            Encoded features [batch, output_dim]
        """
        return self.encoder(x)


class HybridNeuralClassifier(nn.Module):
    """
    Neural network classifier that fuses quantum and classical features.

    Architecture:
        Classical Features -> Encoder -> Embedding
                                        +
        Quantum Features   -> Encoder -> Embedding
                                        |
                                    Fusion Layer
                                        |
                                    Classifier
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.classical_encoder = ClassicalFeatureEncoder(
            config.classical_feature_dim,
            config.hidden_dims,
            config.dropout
        )

        self.quantum_encoder = QuantumFeatureEncoder(
            config.quantum_feature_dim,
            [d // 2 for d in config.hidden_dims],  # Smaller for quantum
            config.dropout
        )

        # Fusion layer
        fusion_dim = config.hidden_dims[-1] + config.hidden_dims[-1] // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Classification head
        self.classifier = nn.Linear(fusion_dim // 2, 2)  # Binary classification

    def forward(
        self,
        classical_features: torch.Tensor,
        quantum_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training/inference.

        Args:
            classical_features: [batch, classical_dim]
            quantum_features: [batch, quantum_dim]

        Returns:
            Logits [batch, 2]
        """
        # Encode features
        classical_emb = self.classical_encoder(classical_features)
        quantum_emb = self.quantum_encoder(quantum_features)

        # Fuse
        combined = torch.cat([classical_emb, quantum_emb], dim=1)
        fused = self.fusion(combined)

        # Classify
        logits = self.classifier(fused)

        return logits

    def predict_proba(
        self,
        classical_features: torch.Tensor,
        quantum_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            classical_features: [batch, classical_dim]
            quantum_features: [batch, quantum_dim]

        Returns:
            Probabilities [batch, 2]
        """
        logits = self.forward(classical_features, quantum_features)
        probs = torch.softmax(logits, dim=1)
        return probs


class QGADSystem(nn.Module):
    """
    Complete Q-GAD system combining quantum feature extraction and classification.

    This is the main model class that orchestrates:
    1. Quantum feature extraction (GBS kernel)
    2. Classical feature processing
    3. Hybrid classification

    Training strategy:
    - Phase 1: Pre-train quantum kernel (unsupervised/supervised)
    - Phase 2: Train hybrid classifier (end-to-end)
    - Phase 3: Fine-tune with tree-based model
    """

    def __init__(
        self,
        gbs_config,
        hybrid_config: HybridConfig,
        use_xgboost_fusion: bool = False
    ):
        super().__init__()

        # Import quantum feature extractor
        from gbs.gbs_kernel import QuantumFeatureExtractor

        self.quantum_extractor = QuantumFeatureExtractor(gbs_config)
        self.hybrid_classifier = HybridNeuralClassifier(hybrid_config)
        self.use_xgboost_fusion = use_xgboost_fusion

        # XGBoost model (trained separately)
        self.xgboost_model = None

        # For alternating training
        self.training_phase = "hybrid"  # 'quantum' or 'hybrid'

    def forward(
        self,
        squeezing_params: torch.Tensor,
        unitary: torch.Tensor,
        classical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            squeezing_params: [batch, n_modes] - Quantum encoding parameters
            unitary: [batch, n_modes, n_modes] - Quantum interferometer
            classical_features: [batch, classical_dim] - Classical features

        Returns:
            Logits [batch, 2]
        """
        # Extract quantum features
        quantum_features = self.quantum_extractor(squeezing_params, unitary)

        # Classify
        logits = self.hybrid_classifier(classical_features, quantum_features)

        return logits

    def train_quantum_kernel(
        self,
        squeezing_params: torch.Tensor,
        unitary: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train only the quantum kernel (alternating training phase).

        Objective: Maximize separation between fraud/normal in quantum feature space.

        Args:
            squeezing_params: Graph encoding parameters
            unitary: Interferometer matrices
            labels: Binary labels
            optimizer: Optimizer for quantum parameters

        Returns:
            Loss and metrics
        """
        self.training_phase = "quantum"

        # Freeze classical classifier
        for param in self.hybrid_classifier.parameters():
            param.requires_grad = False

        # Unfreeze quantum kernel
        for param in self.quantum_extractor.parameters():
            param.requires_grad = True

        # Forward pass
        quantum_features = self.quantum_extractor(squeezing_params, unitary)

        # Contrastive loss: separate fraud vs normal in feature space
        fraud_features = quantum_features[labels == 1]
        normal_features = quantum_features[labels == 0]

        if len(fraud_features) > 0 and len(normal_features) > 0:
            # Minimize intra-class variance, maximize inter-class distance
            fraud_mean = torch.mean(fraud_features, dim=0)
            normal_mean = torch.mean(normal_features, dim=0)

            intra_fraud = torch.mean(torch.square(fraud_features - fraud_mean))
            intra_normal = torch.mean(torch.square(normal_features - normal_mean))
            inter_distance = torch.sum(torch.square(fraud_mean - normal_mean))

            # Loss: minimize intra-class, maximize inter-class
            loss = (intra_fraud + intra_normal) / (inter_distance + 1e-6)
        else:
            # Single class in batch - create dummy loss with gradient
            loss = torch.mean(torch.square(quantum_features)) * 0.0 + 1e-6

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item(), "intra_class": loss.item()}

    def train_hybrid_classifier(
        self,
        squeezing_params: torch.Tensor,
        unitary: torch.Tensor,
        classical_features: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train the hybrid classifier (alternating training phase).

        Args:
            squeezing_params: Graph encoding parameters
            unitary: Interferometer matrices
            classical_features: Classical tabular features
            labels: Binary labels
            optimizer: Optimizer for classifier

        Returns:
            Loss and metrics
        """
        self.training_phase = "hybrid"

        # Freeze quantum kernel
        for param in self.quantum_extractor.parameters():
            param.requires_grad = False

        # Unfreeze classical classifier
        for param in self.hybrid_classifier.parameters():
            param.requires_grad = True

        # Forward pass
        logits = self.forward(squeezing_params, unitary, classical_features)

        # Classification loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }

    def set_xgboost_model(self, model):
        """Set externally trained XGBoost model."""
        self.xgboost_model = model

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "quantum_extractor": self.quantum_extractor.state_dict(),
            "hybrid_classifier": self.hybrid_classifier.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.quantum_extractor.load_state_dict(checkpoint["quantum_extractor"])
        self.hybrid_classifier.load_state_dict(checkpoint["hybrid_classifier"])


class XGBoostFusionClassifier:
    """
    XGBoost classifier that uses quantum features as input.

    This is the final stage of the pipeline:
    1. Extract quantum features (fixed)
    2. Concatenate with classical features
    3. Train XGBoost
    """

    def __init__(
        self,
        quantum_feature_dim: int = 9,
        classical_feature_dim: int = 10,
        **xgb_params
    ):
        self.quantum_feature_dim = quantum_feature_dim
        self.classical_feature_dim = classical_feature_dim
        self.total_feature_dim = quantum_feature_dim + classical_feature_dim

        # Default XGBoost parameters
        default_params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        default_params.update(xgb_params)

        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**default_params)
            self.has_xgboost = True
        except ImportError:
            print("Warning: XGBoost not available, using sklearn's GradientBoosting")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.has_xgboost = False

    def fit(
        self,
        quantum_features: np.ndarray,
        classical_features: np.ndarray,
        labels: np.ndarray,
        validation_data: Optional[Tuple] = None
    ):
        """
        Train XGBoost on combined features.

        Args:
            quantum_features: [n_samples, quantum_dim]
            classical_features: [n_samples, classical_dim]
            labels: [n_samples]
            validation_data: Optional (X_val, y_val) for early stopping
        """
        # Concatenate features
        X = np.concatenate([quantum_features, classical_features], axis=1)

        if validation_data is not None and self.has_xgboost:
            q_val, c_val, y_val = validation_data
            X_val = np.concatenate([q_val, c_val], axis=1)

            self.model.fit(
                X, labels,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X, labels)

    def predict(
        self,
        quantum_features: np.ndarray,
        classical_features: np.ndarray
    ) -> np.ndarray:
        """Predict class labels."""
        X = np.concatenate([quantum_features, classical_features], axis=1)
        return self.model.predict(X)

    def predict_proba(
        self,
        quantum_features: np.ndarray,
        classical_features: np.ndarray
    ) -> np.ndarray:
        """Predict class probabilities."""
        X = np.concatenate([quantum_features, classical_features], axis=1)
        return self.model.predict_proba(X)

    def save_model(self, path: str):
        """Save XGBoost model."""
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load XGBoost model."""
        self.model = joblib.load(path)


if __name__ == "__main__":
    print("Testing Hybrid Classifier...")

    # Create mock data
    batch_size = 32
    quantum_dim = 9
    classical_dim = 10

    quantum_features = torch.randn(batch_size, quantum_dim)
    classical_features = torch.randn(batch_size, classical_dim)
    labels = torch.randint(0, 2, (batch_size,))

    # Test neural classifier
    config = HybridConfig(
        quantum_feature_dim=quantum_dim,
        classical_feature_dim=classical_dim,
        hidden_dims=[32, 16]
    )

    classifier = HybridNeuralClassifier(config)
    logits = classifier(classical_features, quantum_features)

    print(f"Neural classifier output shape: {logits.shape}")

    # Test XGBoost classifier
    xgb_clf = XGBoostFusionClassifier(quantum_dim, classical_dim)

    xgb_clf.fit(
        quantum_features.numpy(),
        classical_features.numpy(),
        labels.numpy()
    )

    probs = xgb_clf.predict_proba(
        quantum_features.numpy(),
        classical_features.numpy()
    )

    print(f"XGBoost output shape: {probs.shape}")
    print(f"XGBoost predictions: {probs[:5]}")

    print("\nHybrid Classifier test completed!")
