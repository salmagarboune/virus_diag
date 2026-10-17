"""Point d'entrée : python main.py   (ou python main.py --cli)"""
import sys
from pathlib import Path

from data.generate_data import generate
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.optimizer import GradientDescent
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor
from utils.preprocessing import Standardizer
from app.api import ClinicalAPI
from app.interface_clinique import run_cli

CSV = Path(__file__).parent / "data" / "patient_data.csv"


def main():
    if not CSV.exists():
        generate(CSV)

    ds = Dataset(CSV).load()
    X_tr, X_te, y_tr, y_te = ds.split()
    scaler = Standardizer().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    d = X_tr.shape[1]

    modeles = {"Régression logistique": LogisticRegression(d),
               "Réseau de neurones": NeuralNetwork(d, hidden=8)}
    predictors = {}

    for nom, modele in modeles.items():
        Trainer(GradientDescent(lr=0.1), epochs=1500).train(modele, X_tr_s, y_tr)
        res = Evaluator().evaluate(modele, X_te_s, y_te)
        print(f"{nom:22s}", {k: round(v, 3) for k, v in res.items()})
        predictors[nom] = ClinicalPredictor(modele, scaler, ds.feature_names)

    patient = {"age": 45, "temperature": 39.2, "toux": 1,
               "fatigue": 1, "saturation_o2": 91.0}
    print("\nPatient test :", patient)
    for nom, p in predictors.items():
        print(f"  {nom:22s} -> {p.diagnose(patient)}")

    print("\nAPI :", ClinicalAPI(predictors["Régression logistique"]).post_diagnose(patient))

    if "--cli" in sys.argv:
        run_cli(predictors["Régression logistique"])


if __name__ == "__main__":
    main()
