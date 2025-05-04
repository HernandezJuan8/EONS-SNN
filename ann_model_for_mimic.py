from datetime import timedelta
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def build_patient_matrix(df, start_time, n_bins=72, bin_hours=1):
    matrix = np.full((n_bins, len(ITEMID_TO_NAME)), np.nan)
    df = df[df['charttime'] >= start_time]
    df = df[df['charttime'] < start_time + timedelta(hours=n_bins * bin_hours)]

    for _, row in df.iterrows():
        time_offset = (row['charttime'] - start_time).total_seconds() / 3600
        bin_idx = int(time_offset // bin_hours)
        if 0 <= bin_idx < n_bins:
            lab_idx = list(ITEMID_TO_NAME.keys()).index(row['itemid'])
            matrix[bin_idx, lab_idx] = row['valuenum']  # last value in bin (you can avg if desired)
    
    return matrix

ITEMID_TO_NAME = {
    50868: "Sodium",
    50882: "Potassium",
    50910: "CK_MB",
    50912: "Creatinine",
    50954: "Troponin_T",
    50952: "BNP",
    51222: "Hemoglobin",
    51301: "WBC"
}

# Load CSVs
diagnoses = pd.read_csv('mimic_iii/DIAGNOSES_ICD.csv')
admissions = pd.read_csv('mimic_iii/ADMISSIONS.csv')
patients = pd.read_csv('mimic_iii/PATIENTS.csv')
labs = pd.read_csv('mimic_iii/LABEVENTS.csv', usecols=[
    'subject_id', 'itemid', 'charttime', 'valuenum', 'valueuom', 'flag'
])
# Filter for heart-related conditions
heart_disease_codes = ['41401', '4280', '4281', '4289', '41071', '42731']
heart_diagnoses = diagnoses[diagnoses['icd9_code'].isin(heart_disease_codes)]

# Get matching subject_ids
heart_patients = set(heart_diagnoses['subject_id'].unique())

labs = labs[labs['itemid'].isin(ITEMID_TO_NAME.keys())]
labs['charttime'] = pd.to_datetime(labs['charttime'])
labs.sort_values(by=['subject_id', 'charttime'], inplace=True)

# === Build patient matrices ===
patient_ids = labs['subject_id'].unique()
X, y = [], []
X_temporal = []

for pid in patient_ids:
    df_pid = labs[labs['subject_id'] == pid]
    if df_pid.empty:
        continue
    start_time = df_pid['charttime'].min()
    mat = build_patient_matrix(df_pid, start_time)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        flat = np.nanmean(mat, axis=0)  # Shape (8,)  -- mean value over time
    if np.isnan(flat).all():
        continue  # skip patients with no valid labs

    X.append(flat)
    X_temporal.append(mat)
    y.append(1 if pid in heart_patients else 0)

X = np.array(X)
X_temporal = np.array(X_temporal)
y = np.array(y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Let's say X is a list of flattened ECG signals, y are 0/1 labels
    accuracys = []
    random_events = np.arange(0,100)
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        accuracys.append(clf.score(X_test, y_test))

    plt.figure()
    plt.title("Random Forest Classifier (Different Train Test Splits)")
    plt.xlabel("Random Seed ID")
    plt.ylabel("Accuracy")

    mean = np.mean(accuracys)
    mean_accuracy = np.full_like(random_events, fill_value=mean, dtype=float)
    plt.plot(random_events, mean_accuracy, label=f"Mean: {mean}")
    plt.plot(random_events, accuracys)
    plt.legend()
    plt.show()