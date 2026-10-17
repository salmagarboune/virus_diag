"""Utilise un modèle DÉJÀ ENTRAÎNÉ pour diagnostiquer un nouveau patient."""
import numpy as np
from core.model import BaseModel


class ClinicalPredictor:
    SEUIL = 0.5

    def __init__(self, model: BaseModel, scaler, feature_names):
        self.model = model              # modèle déjà entraîné (polymorphisme)
        self.scaler = scaler
        self.feature_names = feature_names

    def _to_vector(self, patient_data):
        if isinstance(patient_data, dict):
            patient_data = [patient_data[f] for f in self.feature_names]
        return self.scaler.transform(np.asarray(patient_data, dtype=float))

    def predict_proba(self, patient_data):
        return self.model.predict(self._to_vector(patient_data))

    def diagnose(self, patient_data):
        """Retourne 'Infecté' si prédiction >= 0.5, 'Sain' sinon."""
        return "Infecté" if self.predict_proba(patient_data) >= self.SEUIL else "Sain"
