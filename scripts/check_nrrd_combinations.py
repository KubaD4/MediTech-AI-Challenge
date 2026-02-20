import os
import pandas as pd
from pathlib import Path

root = Path("C:\Users\group1\Desktop\Challenge1")  

records = []
for patient_dir in sorted(root.iterdir()):
    if not patient_dir.is_dir():
        continue
    files = [f.name for f in patient_dir.glob("*.nrrd")]
    has_A = any("_A." in f for f in files)
    has_B = any("_B." in f for f in files)
    has_V = any("_V." in f for f in files)
    records.append({
        "patient": patient_dir.name,
        "A": has_A,
        "B": has_B,
        "V": has_V,
        "combo": "".join([x for x, v in [("A", has_A), ("B", has_B), ("V", has_V)] if v])
    })

df = pd.DataFrame(records)

print("=== Conteggio per combinazione ===")
print(df["combo"].value_counts().to_string())
print(f"\nTotale pazienti: {len(df)}")

df.to_csv("dataset_audit.csv", index=False)
print("\nSalvato dataset_audit.csv")