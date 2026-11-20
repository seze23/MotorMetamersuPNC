"""Exploratory exact blocked tests on cached model circularity proxies.

Five spindle-coefficient groups are blocks; the four checkpoint replicate IDs
within each group are averaged before testing. The tests assume exchangeability
of N/FS/FT labels within each block under the null, and independent blocks.
Those assumptions are not established by the checkpoint archive: these are
conditional model-sensitivity diagnostics, not human population inference.
No movement-frame samples are counted as independent observations.
"""
from pathlib import Path
from itertools import permutations,product
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/soechting_2005'

def omnibus_exact(values):
    values=np.asarray(values,dtype=float)
    centered=values-values.mean(axis=1,keepdims=True)
    statistic=float(np.sum(centered.mean(axis=0)**2))
    arrangements=np.asarray(list(permutations(range(3))))
    choices=np.asarray(list(product(range(6),repeat=len(values))))
    shuffled=np.stack([centered[i][arrangements[choices[:,i]]] for i in range(len(values))],axis=1)
    null=np.sum(shuffled.mean(axis=1)**2,axis=1)
    return statistic,float(np.mean(null>=statistic-1e-15)),len(null)

def sign_flip_exact(values):
    values=np.asarray(values,dtype=float)
    signs=np.asarray(list(product([-1,1],repeat=len(values))))
    observed=float(abs(values.mean()));null=np.abs((signs*values).mean(axis=1))
    return float(values.mean()),float(np.mean(null>=observed-1e-15)),len(null)

def holm_two(pvalues):
    order=np.argsort(pvalues);adjusted=np.zeros(2);last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,(2-rank)*pvalues[index]));adjusted[index]=last
    return adjusted

def main():
    data=pd.read_csv(OUT/'model_pses.csv')
    data['coefficient_group']=data.variant.str.split('_').str[0].astype(int)
    assert len(data)==120 and data.pse.notna().all()
    grouped=data.groupby(['study_experiment','coefficient_group','condition']).agg(mean_pse=('pse','mean'),n_checkpoints=('variant','size')).reset_index()
    assert (grouped.n_checkpoints==4).all()
    grouped.to_csv(OUT/'statistical_group_means.csv',index=False)
    rows=[];contrasts=[]
    for exp in [1,2]:
        individual=data[data.study_experiment==exp].pivot(index='variant',columns='condition',values='pse')
        table=grouped[grouped.study_experiment==exp].pivot(index='coefficient_group',columns='condition',values='mean_pse').loc[:,['N','FS','FT']]
        stat,p,n=omnibus_exact(table.to_numpy())
        mean,base_p,base_n=sign_flip_exact(table.N.to_numpy())
        rows.append(dict(study_experiment=exp,n_coefficient_groups=5,n_checkpoints_per_group=4,omnibus_statistic=stat,speed_condition_p=p,n_label_permutations=n,normal_mean_pse=mean,normal_bias_vs_zero_p=base_p,n_sign_permutations=base_n))
        block_rows=[]
        for condition in ['FS','FT']:
            difference=table[condition]-table.N
            effect,pair_p,perms=sign_flip_exact(difference)
            values=individual[condition]-individual.N
            block_rows.append(dict(study_experiment=exp,contrast=condition+'-N',mean_paired_shift=effect,sd_shift_across_20_variants=values.std(ddof=1),min_variant_shift=values.min(),max_variant_shift=values.max(),n_positive_variants=int((values>0).sum()),n_negative_variants=int((values<0).sum()),paired_sign_flip_p=pair_p,n_sign_permutations=perms))
        adjusted=holm_two([r['paired_sign_flip_p'] for r in block_rows])
        for row,padj in zip(block_rows,adjusted):row['holm_p_two_contrasts']=padj
        contrasts.extend(block_rows)
    omnibus_adjusted=holm_two([r['speed_condition_p'] for r in rows])
    for row,padj in zip(rows,omnibus_adjusted):row['holm_p_if_both_experiments_family']=padj
    pd.DataFrame(rows).to_csv(OUT/'statistical_tests.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'statistical_contrasts.csv',index=False)
    notes=dict(primary_question='Does N/FS/FT change per-variant geometric circularity proxy? Experiment 2 is the main protocol of interest; Experiment 1 is secondary.',unit='five spindle-coefficient group means, each averaging four matched checkpoints',cautions=['These are conditional exploratory model-variant tests, not tests of human perception.','Within-block condition-label exchangeability and independent groups are assumptions, not verified experimental randomization.','Checkpoint archive does not verify independent training seeds; coefficient selections can overlap.','The 20-condition mean trajectory is deterministic and supplies no repeated-trial error distribution.','Tests use per-checkpoint PSEs, not the PSE of the mean physical trajectory; these are different estimators.','PSEs are interpolated from a coarse and restricted alpha grid; grid brackets are numerical resolution, not confidence intervals.','A nonsignificant p value is not an equivalence test or proof of no effect.','Group labels and contrasts were specified before computing these p values; all variants are retained.'])
    (OUT/'statistical_method.json').write_text(json.dumps(notes,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))
if __name__=='__main__':main()
