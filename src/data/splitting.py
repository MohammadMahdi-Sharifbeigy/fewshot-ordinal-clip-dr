"""Patient-aware split helpers shared by build_splits.py and FewShotDataModule.

DeepDRiD has several images per patient (two per eye). A validation/query set
that shares patients with the training images it is used to select would be
optimistic, so every holdout here is grouped by patient.
"""
import numpy as np
import pandas as pd


def patient_ids(df: pd.DataFrame) -> pd.Series:
    """Patient id per row from the manifest's `patient` column.

    Manifests without patient ids (APTOS, MMRDR) treat every image as its own
    group, so the helpers below fall back to plain image-level splits.
    """
    if "patient" in df.columns:
        return df["patient"].astype(str)
    return df["path"].astype(str)


def patient_holdout(df_train: pd.DataFrame, frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out `frac` of the patients, stratified by each patient's highest grade.

    Returns (train_paths, val_paths); no patient appears on both sides.
    """
    df = df_train.assign(_patient=patient_ids(df_train).to_numpy())
    patient_grade = df.groupby("_patient")["grade"].max()
    rng = np.random.default_rng(seed)
    val_patients = []
    for _, group in patient_grade.groupby(patient_grade):
        patients = np.sort(group.index.to_numpy())
        n_val = max(1, int(round(len(patients) * frac)))
        val_patients.extend(rng.choice(patients, size=n_val, replace=False).tolist())
    is_val = df["_patient"].isin(set(val_patients))
    return df.loc[~is_val, "path"].tolist(), df.loc[is_val, "path"].tolist()


def support_order(df_train: pd.DataFrame, grade: int, seed: int) -> list[str]:
    """Seeded sampling order of the train images of one grade.

    Taking the first k gives the k-shot support, so supports are nested across k
    (the 8-shot set contains the 4-shot set) and do not depend on which other k
    values are generated. Images of patients not yet used come first, so a
    support spreads over as many patients as possible.
    """
    df = df_train.assign(_patient=patient_ids(df_train).to_numpy())
    df = df[df["grade"] == grade].sort_values("path")
    rng = np.random.default_rng([seed, grade])
    df = df.iloc[rng.permutation(len(df))]
    nth_of_patient = df.groupby("_patient").cumcount().to_numpy()
    order = np.argsort(nth_of_patient, kind="stable")
    return df["path"].iloc[order].tolist()


def few_shot_split(df_train: pd.DataFrame, k: int, seed: int) -> tuple[list[str], list[str]]:
    """k images per grade as support; the query holds every image of the other patients."""
    support = []
    for grade in sorted(df_train["grade"].unique()):
        order = support_order(df_train, grade, seed)
        if len(order) < k:
            print(f"Warning: grade {grade} has only {len(order)} images for {k}-shot.")
        support.extend(order[:k])
    patients = patient_ids(df_train)
    support_patients = set(patients[df_train["path"].isin(set(support))])
    query = df_train.loc[~patients.isin(support_patients).to_numpy(), "path"].tolist()
    return support, query
