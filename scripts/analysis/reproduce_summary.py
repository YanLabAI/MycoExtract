from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2] / "analysis_outputs" / "reproduction"
def wilson(k,n):
 z=1.959963984540054;d=1+z*z/n;mid=(k/n+z*z/(2*n))/d;h=z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/d
 return [100*(mid-h),100*(mid+h)]
def prf(a):
 t,f,n=a[...,0],a[...,1],a[...,2]
 with np.errstate(divide='ignore',invalid='ignore'):
  return np.stack([t/(t+f),t/(t+n),2*t/(2*t+f+n)],axis=-1)
ledger=pd.read_csv(ROOT/'validation/record_evaluation/per_paper_ledger.csv')
a=ledger.groupby('publication_cluster',sort=False)[['TP','FP','FN']].sum().to_numpy()
tot=a.sum(axis=0);rng=np.random.default_rng(20261011)
boot=a[rng.integers(0,len(a),size=(20000,len(a)))].sum(axis=1)
print('Record counts',dict(zip(['TP','FP','FN'],map(int,tot))))
print('Record metrics and publication-bootstrap 95% intervals',100*prf(tot),np.nanpercentile(100*prf(boot),[2.5,97.5],axis=0))
d=pd.read_csv(ROOT/'validation/document_evaluation/per_publication_document_ledger.csv')
c=d.document_result_using_original_20_10_groups.value_counts();print('Document counts',c.to_dict())
for name,k,n in [('Sensitivity',c.get('TP',0),c.get('TP',0)+c.get('FN',0)),('Specificity',c.get('TN',0),c.get('TN',0)+c.get('FP',0))]:print(name,100*k/n,wilson(k,n))
f=pd.read_csv(ROOT/'validation/field_and_source_audit/six_field_scores.csv')
for _,r in f.iterrows():print('Field',r.field_group,100*prf(np.array([r.TP,r.FP,r.FN],dtype=float)))
n=pd.read_csv(ROOT/'validation/field_and_source_audit/numerical_source_audit.csv')
for name,decision,evaluable in [('Exact numeral','value_exact',None),('Unit','unit_correct','unit_correct'),('Value-unit pair','pair_correct',None),('Direct grounding','direct_source_grounded',None),('Normalized numeral','normalized_value_correct',None),('Normalized pair','normalized_pair_correct',None),('Unsupported under normalization','unsupported_under_normalization_rule',None)]:
 good=n[decision].eq('Yes').astype(int)
 total=n[evaluable].ne('Not evaluated').astype(int) if evaluable else pd.Series(np.ones(len(n),dtype=int),index=n.index)
 tmp=pd.DataFrame({'group':n.publication_cluster,'good':good,'total':total}).groupby('group',sort=False)[['good','total']].sum().to_numpy()
 rng=np.random.default_rng(20261011);sample=tmp[rng.integers(0,len(tmp),size=(20000,len(tmp)))].sum(axis=1)
 ci=np.percentile(100*sample[:,0]/sample[:,1],[2.5,97.5]);print(name,int(good.sum()),int(total.sum()),100*good.sum()/total.sum(),ci)
p=pd.read_csv(ROOT/'reviewer_revision_20261007/nested_all_targets_20261010_oof_predictions.csv')
print('Nested prediction columns',p.columns.tolist())
target_col=next(c for c in ['target','target_name'] if c in p)
y_col=next(c for c in ['observed','y_true','y'] if c in p)
pred_col=next(c for c in ['predicted','y_pred','prediction'] if c in p)
for name,g in p.groupby(target_col):
 y=g[y_col].to_numpy();pred=g[pred_col].to_numpy()
 residual=y-pred; r2=1-np.sum(residual**2)/np.sum((y-y.mean())**2)
 rho=np.corrcoef(pd.Series(y).rank(),pd.Series(pred).rank())[0,1]
 print('Nested',name,len(g),r2,np.sqrt(np.mean(residual**2)),np.mean(np.abs(residual)),rho)
