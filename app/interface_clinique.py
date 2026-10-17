"""Interface clinique en ligne de commande."""


def run_cli(predictor):
    print("=== Interface clinique : diagnostic patient ===")
    while True:
        patient = {}
        for nom in predictor.feature_names:
            patient[nom] = float(input(f"{nom} : "))
        proba = predictor.predict_proba(patient)
        print(f"-> Diagnostic : {predictor.diagnose(patient)} (p = {proba:.2f})")
        if input("Autre patient ? (o/n) ").strip().lower() != "o":
            break
