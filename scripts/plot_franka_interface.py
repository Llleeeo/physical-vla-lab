#!/usr/bin/env python3
"""Plot the recorded standalone interface checks without running Isaac."""
import argparse
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/capstone_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run_directory',type=Path)
a=p.parse_args()
with np.load(a.run_directory/'trace.npz') as h:data={k:h[k] for k in h.files}
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for name in ['move_x','move_y','move_z']:
 m=data['case']==name
 axes[0,0].plot(data['time_s'][m],data['position_error_m'][m]*1000,label=name)
axes[0,0].axvspan(2.8,3,color='green',alpha=.12,label='final acceptance window')
axes[0,0].set(title='3 cm movement: transient and settling error',xlabel='Time (s)',ylabel='Position error (mm)')
axes[0,0].legend(fontsize=8)
m=data['case']=='circle'
if np.any(m):
 origin=data['action'][m][0,:2]
 for xy,label in [(data['action'][m][:,:2],'command'),(data['measured_ee_position_base'][m][:,:2],'measured')]:
  axes[0,1].plot((xy[:,0]-origin[0])*1000,(xy[:,1]-origin[1])*1000,label=label)
 axes[0,1].set_aspect('equal')
axes[0,1].set(title='1 cm radius, 6 s circle',xlabel='Base X displacement (mm)',ylabel='Base Y displacement (mm)')
axes[0,1].legend()
m=data['case']=='rotate_z'
if np.any(m):
 goal=Rotation.from_euler('xyz',data['action'][m][:,3:6]);base=goal[0]
 for rotation,label in [(goal,'command'),(Rotation.from_quat(data['measured_ee_quaternion_xyzw'][m]),'measured')]:
  axes[1,0].plot(data['time_s'][m],(rotation*base.inv()).as_rotvec()[:,2]*180/np.pi,label=label)
axes[1,0].set(title='10 degree rotation about base Z',xlabel='Time (s)',ylabel='Rotation (degrees)');axes[1,0].legend()
for case,offset in [('close',0),('open',2)]:
 m=data['case']==case
 if np.any(m):
  time=data['time_s'][m]+offset
  axes[1,1].plot(time,data['joint_pos'][m][:,7]*1000,label=case+' measured')
  axes[1,1].plot(time,np.where(data['action'][m][:,6]>0,40,0),'k--',alpha=.5)
axes[1,1].set(title='Finger joint travel (one finger shown)',xlabel='Time (s): close then open',ylabel='Joint position (mm)');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.25)
fig.suptitle('Standalone Franka interface: 60 Hz commands / 180 Hz servo and physics')
fig.savefig(a.run_directory/'interface_checks.png',dpi=160)
print(a.run_directory/'interface_checks.png')
