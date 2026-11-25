"""Reproduce all 50 reported predictions from frozen training/candidate features.
No literature retrieval, LLM calls or protein embedding download is required.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--modeling-data', type=Path, default=ROOT / 'data/modeling')
base = parser.parse_args().modeling_data.resolve()
HERE = base / 'retrospective'
df = pd.read_excel(base / 'kinetic_modeling.xlsx', sheet_name='Measurements')
X = np.load(base / 'feature_matrix.npz')['X']
test=np.load(HERE/'external_features.npz')['X']
candidates=json.loads((HERE/'candidates.json').read_text(encoding='utf8'))
assert X.shape==(72,1514) and test.shape==(22,1514) and len(candidates)==22
expected=pd.read_csv(HERE/'all_predictions.csv')
results=[]
for col,target in [('log10_kcat_s','kcat_s'),('log10_Km_uM','Km_uM'),('log10_kcat_Km_M_s','kcat_Km_M_s')]:
    valid=df[col].notna().to_numpy();y=df.loc[valid,col].to_numpy(float)
    model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    model.fit(X[valid],y);pred=model.predict(test);mean=float(y.mean())
    for c,value in zip(candidates,pred):
        if target not in c['values']:continue
        results.append({'enzyme':c['enzyme'],'target':target,'predicted_log10':float(value),'baseline_log10':mean})
out=pd.DataFrame(results);merged=expected.merge(out,on=['enzyme','target'],suffixes=('_reported','_recomputed'),validate='one_to_one')
assert len(out)==len(merged)==50
error=np.abs(merged.predicted_log10_reported-merged.predicted_log10_recomputed)
baseline_error=np.abs(np.log10(merged.baseline_predicted)-merged.baseline_log10)
assert float(error.max())<1e-4,float(error.max())
assert float(baseline_error.max())<1e-10,float(baseline_error.max())
recomputed_fold_error=10.0**np.abs(merged.predicted_log10_recomputed-np.log10(merged.observed))
assert np.array_equal(recomputed_fold_error.le(2).to_numpy(),merged.fold_error.le(2).to_numpy())
dest = ROOT / 'analysis_outputs/reproduction/retrospective'
dest.mkdir(parents=True, exist_ok=True)
out.to_csv(dest/'all_predictions_recomputed.csv',index=False)
report={'measurements':50,'enzymes_or_variants':22,'max_absolute_log10_prediction_difference':float(error.max()),'max_absolute_log10_baseline_difference':float(baseline_error.max()),'tolerance_log10':1e-4,'all_values_verified_within_tolerance':True,'twofold_classifications_unchanged':True,'same_log10_predictions_to_three_decimals':int((np.round(merged.predicted_log10_reported,3)==np.round(merged.predicted_log10_recomputed,3)).sum())}
(dest/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
