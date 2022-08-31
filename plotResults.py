'''
    This script plots results of the simulations. You should specify which
    'case' you want to plot in the list 'cases' below.
'''

# %% Import packages.
import os
import numpy as np
import matplotlib.pyplot as plt  

# %% User inputs
cases = ['0', '1', '2']

# %% Paths
pathMain = os.getcwd()
# Load results
pathTrajectories = os.path.join(pathMain, 'Results')
optimaltrajectories = np.load(os.path.join(pathTrajectories, 
                                           'optimalTrajectories.npy'),
                              allow_pickle=True).item()
# Load experimental data
pathData = os.path.join(pathMain, 'OpenSimModel', 'Hamner_modified')
experimentalData = np.load(os.path.join(pathData, 'experimentalData.npy'),
                           allow_pickle=True).item()
subject = 'new_model'

# %% Joint positions.
joints = optimaltrajectories[cases[0]]['joints']
jointToPlot = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 
               'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 
               'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 
               'knee_angle_r',  'ankle_angle_r', 
               'subtalar_angle_r', 'mtp_angle_r', 
               'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
               'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r']
from utilities import getJointIndices
idxJointsToPlot = getJointIndices(joints, jointToPlot)
NJointsToPlot = len(jointToPlot)    
fig, axs = plt.subplots(4, 6, sharex=True)    
fig.suptitle('Joint positions')
for i, ax in enumerate(axs.flat):
    if i < NJointsToPlot:
        color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
        plotExperimental = True
        for case in cases:
            c_joints = optimaltrajectories[case]['joints']            
            if not jointToPlot[i] in c_joints:
                continue            
            c_joint_idx = c_joints.index(jointToPlot[i])             
            ax.plot(optimaltrajectories[case]['GC_percent'],
                    optimaltrajectories[case]['coordinate_values'][c_joint_idx:c_joint_idx+1, :].T, c=next(color), label='case_' + case)
            if plotExperimental:
                ax.fill_between(experimentalData[subject]["kinematics"]["positions"]["GC_percent"],
                            experimentalData[subject]["kinematics"]["positions"]["mean"][jointToPlot[i]] + 2*experimentalData[subject]["kinematics"]["positions"]["std"][jointToPlot[i]],
                            experimentalData[subject]["kinematics"]["positions"]["mean"][jointToPlot[i]] - 2*experimentalData[subject]["kinematics"]["positions"]["std"][jointToPlot[i]],
                            facecolor='grey', alpha=0.4)
                plotExperimental = False
        ax.set_title(joints[idxJointsToPlot[i]])
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
plt.setp(axs[:, 0], ylabel='(deg or m)')
fig.align_ylabels()

# %% Muscle activations.
muscles = optimaltrajectories[cases[0]]['muscles']
musclesToPlot = ['glut_med1_r', 'glut_med2_r', 'glut_med3_r', 'glut_min1_r', 
                  'glut_min2_r', 'glut_min3_r', 'semimem_r', 'semiten_r', 
                  'bifemlh_r', 'bifemsh_r', 'sar_r', 'add_long_r', 'add_brev_r',
                  'add_mag1_r', 'add_mag2_r', 'add_mag3_r', 'tfl_r', 'pect_r',
                  'grac_r', 'glut_max1_r', 'glut_max2_r', 'glut_max3_r',
                  'iliacus_r', 'psoas_r', 'quad_fem_r', 'gem_r', 'peri_r',
                  'rect_fem_r', 'vas_med_r', 'vas_int_r', 'vas_lat_r',
                  'med_gas_r', 'lat_gas_r', 'soleus_r', 'tib_post_r',
                  'flex_dig_r', 'flex_hal_r', 'tib_ant_r', 'per_brev_r',
                  'per_long_r', 'per_tert_r', 'ext_dig_r', 'ext_hal_r',
                  'ercspn_r', 'intobl_r', 'extobl_r']
mappingEMG = {'glut_med1_r': 'GluMed_r', 
              'glut_med2_r': 'GluMed_r', 
              'glut_med3_r': 'GluMed_r',
              'semimem_r': 'HamM_r',
              'semiten_r': 'HamM_r',
              'bifemlh_r': 'HamL_r',
              'bifemsh_r': 'HamL_r',
              'add_long_r': 'AddL_r',
              'tfl_r': 'TFL_r',
              'rect_fem_r': 'RF_r',
              'vas_med_r': 'VM_r',
              'vas_int_r': 'VL_r',
              'vas_lat_r': 'VL_r',
              'med_gas_r': 'GM_r',
              'lat_gas_r': 'GL_r',
              'soleus_r': 'Sol_r',
              'tib_ant_r': 'TA_r',
              'per_brev_r': 'PerB_l',
              'per_long_r': 'PerL_l',
              'glut_med1_l': 'GluMed_l', 
              'glut_med2_l': 'GluMed_l', 
              'glut_med3_l': 'GluMed_l',
              'semimem_l': 'HamM_l',
              'semiten_l': 'HamM_l',
              'bifemlh_l': 'HamL_l',
              'bifemsh_l': 'HamL_l',
              'add_long_l': 'AddL_l',
              'tfl_l': 'TFL_l',
              'rect_fem_l': 'RF_l',
              'vas_med_l': 'VM_l',
              'vas_int_l': 'VL_l',
              'vas_lat_l': 'VL_l',
              'med_gas_l': 'GM_l',
              'lat_gas_l': 'GL_l',
              'soleus_l': 'Sol_l',
              'tib_ant_l': 'TA_l',
              'per_brev_l': 'PerB_l',
              'per_long_l': 'PerL_l'}
NMusclesToPlot = len(musclesToPlot)
idxMusclesToPlot = getJointIndices(muscles, musclesToPlot)
fig, axs = plt.subplots(8, 6, sharex=True)    
fig.suptitle('Muscle activations')
for i, ax in enumerate(axs.flat):
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
    if i < NMusclesToPlot:
        plotExperimental = True
        for case in cases:
            ax.plot(optimaltrajectories[case]['GC_percent'],
                    optimaltrajectories[case]['muscle_activations'][idxMusclesToPlot[i]:idxMusclesToPlot[i]+1, :].T, c=next(color), label='case_' + case)            
            if musclesToPlot[i] in mappingEMG and plotExperimental:                
                # Normalize EMG such that peak mean EMG = peak activation             
                exp_mean = experimentalData[subject]["EMG"]["mean"][mappingEMG[musclesToPlot[i]]]
                exp_mean_peak = np.max(exp_mean)
                sim = optimaltrajectories[case]['muscle_activations'][idxMusclesToPlot[i], :].T
                sim_peak = np.max(sim)
                scaling_emg = sim_peak / exp_mean_peak
                ax.fill_between(experimentalData[subject]["EMG"]["GC_percent"],
                        experimentalData[subject]["EMG"]["mean"][mappingEMG[musclesToPlot[i]]] * scaling_emg + 2*experimentalData[subject]["EMG"]["std"][mappingEMG[musclesToPlot[i]]] * scaling_emg,
                        experimentalData[subject]["EMG"]["mean"][mappingEMG[musclesToPlot[i]]] * scaling_emg - 2*experimentalData[subject]["EMG"]["std"][mappingEMG[musclesToPlot[i]]] * scaling_emg,
                        facecolor='grey', alpha=0.4)
                plotExperimental = False    
        ax.set_title(muscles[idxMusclesToPlot[i]])
        ax.set_ylim((0,1))
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
plt.setp(axs[:, 0], ylabel='(-)')
fig.align_ylabels()

# %% Joint torques.
fig, axs = plt.subplots(4, 6, sharex=True)    
fig.suptitle('Joint kinetics') 
for i, ax in enumerate(axs.flat):
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
    if i < NJointsToPlot:
        plotExperimental = True
        for case in cases:
            c_joints = optimaltrajectories[case]['joints']
            if not jointToPlot[i] in c_joints:
                continue
            c_joint_idx = c_joints.index(jointToPlot[i]) 
            ax.plot(optimaltrajectories[case]['GC_percent'],
                    optimaltrajectories[case]['joint_torques'][c_joint_idx:c_joint_idx+1, :].T, c=next(color), label='case_' + case)
            if plotExperimental:
                ax.fill_between(experimentalData[subject]["kinetics"]["GC_percent"],
                            experimentalData[subject]["kinetics"]["mean"][jointToPlot[i]] + 2*experimentalData[subject]["kinetics"]["std"][jointToPlot[i]],
                            experimentalData[subject]["kinetics"]["mean"][jointToPlot[i]] - 2*experimentalData[subject]["kinetics"]["std"][jointToPlot[i]],
                            facecolor='grey', alpha=0.4)
        ax.set_title(joints[idxJointsToPlot[i]])
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
plt.setp(axs[:, 0], ylabel='(Nm)')
fig.align_ylabels()

# %% Ground reaction forces.
GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
GRFToPlot = ['GRF_x_r', 'GRF_y_r', 'GRF_z_r', 'GRF_x_l','GRF_y_l', 'GRF_z_l']
NGRFToPlot = len(GRFToPlot)
idxGRFToPlot = getJointIndices(GRF_labels, GRFToPlot)
fig, axs = plt.subplots(2, 3, sharex=True)    
fig.suptitle('Ground reaction forces')
for i, ax in enumerate(axs.flat):
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
    plotExperimental = True
    for case in cases:
        ax.plot(optimaltrajectories[case]['GC_percent'],
                optimaltrajectories[case]['GRF'][idxGRFToPlot[i]:idxGRFToPlot[i]+1, :].T, c=next(color), label='case_' + case) 
        if plotExperimental:
            ax.fill_between(experimentalData[subject]["GRF"]["GC_percent"],
                            experimentalData[subject]["GRF"]["mean"][GRFToPlot[i]] + 2*experimentalData[subject]["GRF"]["std"][GRFToPlot[i]],
                            experimentalData[subject]["GRF"]["mean"][GRFToPlot[i]] - 2*experimentalData[subject]["GRF"]["std"][GRFToPlot[i]],
                            facecolor='grey', alpha=0.4)
    ax.set_title(GRF_labels[idxGRFToPlot[i]])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
plt.setp(axs[:, 0], ylabel='(N)')
fig.align_ylabels()

# %% Metabolic cost and cost function value.
fig, (ax1, ax2) = plt.subplots(1, 2)
color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
for count, case in enumerate(cases):
    c_c = next(color)
    ax1.scatter(count, optimaltrajectories[case]["COT"], s=80, color=c_c)
    ax2.scatter(count, optimaltrajectories[case]["objective"], s=80, color=c_c)
ax1.set_title("Cost of Transport")
ax1.set_ylabel("(J/Kg/m)")    
ax2.set_title("Optimal cost value")
ax2.set_ylabel("()")
x_locations = np.linspace(0, len(cases)-1, len(cases))
ax1.set_xticks(x_locations)
xticklabels = ["Case_" + case for case in cases]
ax1.set_xticklabels(xticklabels)
ax2.set_xticks(x_locations)
ax2.set_xticklabels(xticklabels)

# %% Muscle fiber lengths.
NMusclesToPlot = len(musclesToPlot)
idxMusclesToPlot = getJointIndices(muscles, musclesToPlot)
fig, axs = plt.subplots(8, 6, sharex=True)    
fig.suptitle('Normalized muscle fiber lengths')
for i, ax in enumerate(axs.flat):
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
    if i < NMusclesToPlot:
        for case in cases:
            ax.plot(optimaltrajectories[case]['GC_percent'],
                    optimaltrajectories[case]['norm_fiber_lengths'][idxMusclesToPlot[i]:idxMusclesToPlot[i]+1, :].T, c=next(color), label='case_' + case)
        ax.set_title(muscles[idxMusclesToPlot[i]])
        ax.set_ylim((0,2))
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
plt.setp(axs[:, 0], ylabel='(-)')
fig.align_ylabels()

# %% Muscle fiber velocities.
NMusclesToPlot = len(musclesToPlot)
idxMusclesToPlot = getJointIndices(muscles, musclesToPlot)
fig, axs = plt.subplots(8, 6, sharex=True)    
fig.suptitle('Muscle fiber velocities')
for i, ax in enumerate(axs.flat):
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
    if i < NMusclesToPlot:
        for case in cases:
            ax.plot(optimaltrajectories[case]['GC_percent'],
                    optimaltrajectories[case]['fiber_velocity'][idxMusclesToPlot[i]:idxMusclesToPlot[i]+1, :].T, c=next(color), label='case_' + case)
        ax.set_title(muscles[idxMusclesToPlot[i]])
        ax.set_ylim((-1,1))
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
plt.setp(axs[:, 0], ylabel='(-)')
fig.align_ylabels()

# %% Cost terms.
fig, ((ax11, ax12, ax13), (ax21, ax22, ax23), (ax31, ax32, ax33)) = plt.subplots(3, 3)
color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases)))) 
for count, case in enumerate(cases):
    c_c = next(color)
    ax11.scatter(count, optimaltrajectories[case]["objective_terms"]["metabolicEnergyRateTerm"], s=80, color=c_c)
    ax12.scatter(count, optimaltrajectories[case]["objective_terms"]["activationTerm"], s=80, color=c_c)
    ax13.scatter(count, optimaltrajectories[case]["objective_terms"]["armExcitationTerm"], s=80, color=c_c)
    ax21.scatter(count, optimaltrajectories[case]["objective_terms"]["jointAccelerationTerm"], s=80, color=c_c)
    ax22.scatter(count, optimaltrajectories[case]["objective_terms"]["passiveTorqueTerm"], s=80, color=c_c)
    ax23.scatter(count, optimaltrajectories[case]["objective_terms"]["activationDtTerm"], s=80, color=c_c)
    ax31.scatter(count, optimaltrajectories[case]["objective_terms"]["forceDtTerm"], s=80, color=c_c)
    ax32.scatter(count, optimaltrajectories[case]["objective_terms"]["armAccelerationTerm"], s=80, color=c_c)
ax11.set_title("metabolicEnergyRateTerm")
ax11.set_ylabel("(J/Kg/m)")    
ax12.set_title("activationTerm")
ax12.set_ylabel("()")
ax13.set_title("armExcitationTerm")
ax13.set_ylabel("()")
ax21.set_title("jointAccelerationTerm")
ax21.set_ylabel("()")
ax22.set_title("passiveJointTorqueTerm")
ax22.set_ylabel("()")
ax23.set_title("activationDtTerm")
ax23.set_ylabel("()")
ax31.set_title("forceDtTerm")
ax31.set_ylabel("()")
ax32.set_title("armAccelerationTerm")
ax32.set_ylabel("()")
x_locations = np.linspace(0, len(cases)-1, len(cases))
ax11.set_xticks(x_locations)
xticklabels = ["Case_" + case for case in cases]
ax11.set_xticklabels(xticklabels)
ax12.set_xticks(x_locations)
ax12.set_xticklabels(xticklabels)
ax13.set_xticks(x_locations)
ax13.set_xticklabels(xticklabels)
ax21.set_xticks(x_locations)
ax21.set_xticklabels(xticklabels)
ax22.set_xticks(x_locations)
ax22.set_xticklabels(xticklabels)
ax23.set_xticks(x_locations)
ax23.set_xticklabels(xticklabels)
ax31.set_xticks(x_locations)
ax31.set_xticklabels(xticklabels)
ax32.set_xticks(x_locations)
ax32.set_xticklabels(xticklabels)
