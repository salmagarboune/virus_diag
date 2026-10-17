"""Génère le fichier synthétique patient_data.csv."""
import numpy as np
import pandas as pd


def generate(path, n=500, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(5, 90, n)
    temperature = np.round(rng.normal(37.5, 1.0, n), 1)
    toux = rng.integers(0, 2, n)
    fatigue = rng.integers(0, 2, n)
    saturation = np.round(np.clip(rng.normal(96, 3, n), 80, 100), 1)

    score = (0.9 * (temperature - 37.5) + 1.2 * toux + 0.8 * fatigue
             - 0.35 * (saturation - 96) + 0.01 * (age - 45) - 0.5)
    proba = 1 / (1 + np.exp(-score))
    infecte = (rng.random(n) < proba).astype(int)

    df = pd.DataFrame({"age": age, "temperature": temperature, "toux": toux,
                       "fatigue": fatigue, "saturation_o2": saturation,
                       "infecte": infecte})
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    generate("patient_data.csv")
