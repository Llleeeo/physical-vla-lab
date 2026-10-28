#!/usr/bin/env python3
"""CPU figures for the recorded-observation prediction diagnostic."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);root=parser.parse_args().output
    with h5py.File(root/'predictions.hdf5','r') as f:
        t=f['timestamp_s'][:];target=f['target_absolute'][:,0]
        fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
        for j,label in enumerate(['x','y','z']):
            ax=axes[j,0];ax.plot(t,target[:,j],color='black',label='expert target')
            for name in ['pretrained','finetuned']:ax.plot(t,f[name+'/predicted_absolute'][:,0,j],label=name,alpha=.85)
            ax.set_ylabel(label+' (m)');ax.grid(alpha=.2)
        for name in ['pretrained','finetuned']:
            axes[0,1].plot(t,f[name+'/position_error_mm'][:,0],label=name)
            axes[1,1].plot(t,f[name+'/orientation_error_deg'][:,0],label=name)
            axes[2,1].plot(t,f[name+'/predicted_absolute'][:,0,6],label=name,alpha=.8)
        axes[0,1].set_ylabel('position error (mm)');axes[1,1].set_ylabel('SO(3) orientation error (deg)')
        axes[2,1].step(t,target[:,6],where='post',color='black',label='expert target')
        axes[2,1].axhline(0,color='grey',linestyle='--');axes[2,1].set_ylabel('gripper raw (+open / -close)')
        for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('recorded source time (s)')
        fig.suptitle('First action of each predicted chunk on RECORDED observations (no simulation rollout)')
        fig.tight_layout();fig.savefig(root/'first_action_comparison.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,4));window=slice(106,121)
        ax.step(np.arange(len(t))[window],target[window,6],where='mid',color='black',label='expert target')
        for name in ['pretrained','finetuned']:
            raw=f[name+'/predicted_absolute'][:,0,6]
            ax.plot(np.arange(len(t))[window],raw[window],marker='o',label=name)
        ax.axhline(0,color='grey',linestyle='--');ax.set_xlabel('source control frame');ax.set_ylabel('gripper command')
        ax.set_title('Gripper transitions 112 / 113 / 114: raw predictions, threshold=0')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'gripper_transition_detail.png',dpi=150);plt.close(fig)
    print('FIGURES',root/'first_action_comparison.png',root/'gripper_transition_detail.png')

if __name__=='__main__':main()
