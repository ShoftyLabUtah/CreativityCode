'''

SVM decoder, but on a single contact level. To make sure poor DAN decoding not just due to low number of contacts. 

+ mixed effects model + OLS

+ leave one subject out robustness check

'''


import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

contact_level_results = []

contrast = input("Compare AUT to BAT or OCT? ")

networks = ['DMN', 'DorsalAttention']

for network in networks:
    print(f"Running network: {network}")
    
    if contrast == 'BAT':
        all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
        print('BAT')
    if contrast == 'OCT':
        ###FOR AUT VS OCT:
        all_dat = pd.read_csv('pkl_files/store_neural_data_list_all_trialtypes_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
        print('OCT')

    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    ###FOR AUT VS OCT:
    if contast == 'OCT':
        neural_data_list_fixed_AUC_OCT = [neural_data.iloc[np.where((neural_data.index.get_level_values('trial_type')=='AUT') | (neural_data.index.get_level_values('trial_type')=='OCT'))[0]] for neural_data in cleaned_dfs]
        cleaned_dfs = neural_data_list_fixed_AUC_OCT

    for patient_df in cleaned_dfs:
        patient = patient_df.index.get_level_values('patient').unique()[0]

        if patient in ['UIC202420', 'UIC202422']:
            continue

        print(patient)

        # columns are MultiIndex: freq_band x channel
        channels = patient_df.columns.get_level_values('channel').unique()

        for channel in channels:

            # Select one contact/channel across all frequency bands
            try:
                one_contact = patient_df.xs(channel, axis=1, level='channel')
            except KeyError:
                continue

            # one_contact index:
            # patient, trial_idx, trial_type, time
            #
            # one_contact columns:
            # frequency bands

            # Convert to trial x features
            # Features become frequency_band x time
            X_df = one_contact.unstack('time')

            # Drop trials with missing data
            X_df = X_df.dropna(axis=0)

            if X_df.shape[0] < 8:
                continue

            y = X_df.index.get_level_values('trial_type')

            # Require at least 2 classes
            if len(np.unique(y)) < 2:
                continue

            # Require enough samples per class for CV
            class_counts = pd.Series(y).value_counts()
            if class_counts.min() < 4:
                continue

            X = X_df.values

            le = LabelEncoder()
            y_num = le.fit_transform(y)

            if contrast == 'BAT': #AUT vs BAT
                clf = SVC(C=1, kernel='linear', tol=1e-6)
            else: #AUT vs OCT
                clf = LinearDiscriminantAnalysis()

            cv = StratifiedKFold(
                n_splits=4,
                shuffle=True,
                random_state=0
            )

            y_true_all = []
            y_pred_all = []
            decision_all = []

            for train_idx, test_idx in cv.split(X, y_num):
                clf.fit(X[train_idx], y_num[train_idx])

                y_pred = clf.predict(X[test_idx])
                decision = clf.decision_function(X[test_idx])

                y_true_all.extend(y_num[test_idx])
                y_pred_all.extend(y_pred)
                decision_all.extend(decision)

            bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)

            # AUC only works cleanly for binary classification
            if len(np.unique(y_num)) == 2:
                auc = roc_auc_score(y_true_all, decision_all)
            else:
                auc = np.nan

            contact_level_results.append({
                'patient': patient,
                'network': network,
                'channel': channel,
                'n_trials': X_df.shape[0],
                'balanced_accuracy': bal_acc,
                'auc': auc
            })

contact_df = pd.DataFrame(contact_level_results)

import statsmodels.formula.api as smf
df = contact_df.copy()
df["is_DAN"] = df["network"].isin(["DorsalAttention"]).astype(int)


print(df["network"].value_counts())
print(df["is_DAN"].value_counts())
print(pd.crosstab(df["patient"], df["is_DAN"]))


####MIXED EFFECTS MODEL: 
df = df.dropna(subset=["auc", "is_DAN", "patient"]).copy()

model = smf.mixedlm(
    "auc ~ is_DAN",
    data=df,
    groups=df["patient"]
)

for method in ["powell", "nm", "bfgs", "cg", "lbfgs"]:
    try:
        res = model.fit(method=method, reml=False, maxiter=2000, disp=False)
        print(f"\nSucceeded with {method}")
        print(res.summary())
        break
    except Exception as e:
        print(f"\nFailed with {method}: {e}")

#####ORDINARY LEAST SQUARES MODEL
ols = smf.ols(
    "auc ~ is_DAN",
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["patient"]}
)

print(ols.summary())

print("DAN coefficient:", ols.params["is_DAN"])
print("Clustered p-value:", ols.pvalues["is_DAN"])
print("95% CI:", ols.conf_int().loc["is_DAN"].tolist())




df["network"] = pd.Categorical(
    df["network"],
    categories=["DMN"] + [n for n in sorted(df["network"].unique()) if n != "DMN"],
    ordered=False
)

ols_network = smf.ols(
    "auc ~ C(network, Treatment(reference='DMN'))",
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["patient"]}
)


#####LEAVE ONE SUBJECT OUR ROBUSTNESS CHECK (i.e. is single-contact DAN decoding  driven by a single participant?)
dan_subjects = df.loc[df["is_DAN"] == 1, "patient"].unique()

loo_results = []

for held_out in dan_subjects:
    tmp = df[df["patient"] != held_out].copy()

    fit = smf.ols(
        "auc ~ is_DAN",
        data=tmp
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": tmp["patient"]}
    )

    loo_results.append({
        "held_out_patient": held_out,
        "dan_coef": fit.params.get("is_DAN", np.nan),
        "dan_pvalue": fit.pvalues.get("is_DAN", np.nan),
        "ci_low": fit.conf_int().loc["is_DAN", 0],
        "ci_high": fit.conf_int().loc["is_DAN", 1],
        "n_DAN_contacts_remaining": int(tmp["is_DAN"].sum()),
        "n_DAN_subjects_remaining": tmp.loc[tmp["is_DAN"] == 1, "patient"].nunique()
    })

loo_df = pd.DataFrame(loo_results)
print(loo_df)