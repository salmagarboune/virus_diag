"""API simulée (sans serveur) : reçoit un JSON-dict, renvoie un JSON-dict."""


class ClinicalAPI:
    def __init__(self, predictor):
        self.predictor = predictor

    def post_diagnose(self, payload: dict) -> dict:
        manquants = [f for f in self.predictor.feature_names if f not in payload]
        if manquants:
            return {"status": 400, "erreur": f"Champs manquants : {manquants}"}
        proba = self.predictor.predict_proba(payload)
        return {"status": 200, "diagnostic": self.predictor.diagnose(payload),
                "probabilite": round(proba, 3)}
