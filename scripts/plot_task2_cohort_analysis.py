#!/usr/bin/env python3
"""Plot CPU audit curves using the existing rendering environment."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('analysis_dir',type=Path);a=p.parse_args()
data=json.loads((a.analysis_dir/'curves.json').read_text());rows=data['episodes']
fig,axes=plt.subplots(len(rows),2,figsize=(14,22))
for row,ax in zip(rows,axes):
 t=row['time'];ax[0].plot(t,row['position'],label='model final');ax[0].plot(t,row['hold_position'],label='hold measured pose',alpha=.8)
 ax[0].set(title=row['episode']+' position error',ylabel='mm');ax[0].legend(fontsize=8)
 ax[1].step(t,row['expert_gripper'],where='post',label='expert');ax[1].step(t,row['model_gripper'],where='post',label='model sign',alpha=.7)
 ax[1].set(title=row['episode']+' gripper',yticks=[0,1],yticklabels=['closed','open']);ax[1].legend(fontsize=8)
for ax in axes[-1]:ax.set_xlabel('recorded time (s)')
fig.tight_layout();fig.savefig(a.analysis_dir/'per_episode_position_gripper.png',dpi=130);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));history=data['flow_history']
for split in ['train','validation']:
 ax.plot([x['step'] for x in history],[x[split+'_fixed_flow_mse'] for x in history],marker='o',label=split)
ax.set(xlabel='optimizer steps',ylabel='fixed probe valid 7D flow MSE',yscale='log');ax.legend();fig.tight_layout();fig.savefig(a.analysis_dir/'flow_history.png',dpi=150);plt.close(fig)
print('FIGURES',a.analysis_dir)
