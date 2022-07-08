'''
    This script extracts experimental data from different files. The data
    are then interpolated over a gait cycle and saved in a .npy file. The
    experimental data can also be plotted.
    
    This script is a little messy, sorry about that.
'''

# %% Import packages.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

from utilities import getGRF, getInitialContact, interpolateDataFrame
from utilities import getIK, getID, getFromStorage

# %% User inputs.
subject = "new_model"
saveExperimentalData = True
plotData = True

# %% Data processing.
trials = {}
trials[subject] = {}
trials[subject]["names"] = ['gait_61', 'gait_63', 'gait_64', 'gait_14',
                            'gait_15', 'gait_23', 'gait_25', 'gait_27',
                            'gait_60', 'gait_65']
# timeIC2 indicates the first initial contact after the force plate, this
# is visually extracted from OpenSim.
trials[subject]["timeIC2"] = [2.87, 1.86, 2.14, 4.27, 3.2, 2.41, 2.2, 2.92, 
                              2.26, 2.1]

# Headers commonly used in .mot files. 
headers = {}
headers["GRF"] = {}
headers["GRF"]["right"] = ["1_ground_force_vx", "1_ground_force_vy", 
                           "1_ground_force_vz"]
headers["GRF"]["left"] = ["ground_force_vx", "ground_force_vy", 
                          "ground_force_vz"]
headers["GRF"]["all"] = headers["GRF"]["right"] + headers["GRF"]["left"] 
# Adjusted headers to avoid confusion.
headers["GRF_adj"] = {}
headers["GRF_adj"]["right"] = ['GRF_x_r', 'GRF_y_r', 'GRF_z_r']
headers["GRF_adj"]["left"] = ['GRF_x_l','GRF_y_l', 'GRF_z_l']
headers["GRF_adj"]["all"] = (headers["GRF_adj"]["right"] + 
                             headers["GRF_adj"]["left"])
nGRFS = len(headers["GRF"]["all"])
# List of joints.
joints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 
          'pelvis_ty', 'pelvis_tz', 'hip_flexion_l', 'hip_adduction_l', 
          'hip_rotation_l', 'hip_flexion_r', 'hip_adduction_r', 
          'hip_rotation_r', 'knee_angle_l', 'knee_angle_r',
          'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l', 
          'subtalar_angle_r', 'mtp_angle_l', 'mtp_angle_r', 
          'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 
          'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'arm_flex_r', 
          'arm_add_r', 'arm_rot_r', 'elbow_flex_l', 'elbow_flex_r']
nJoints = len(joints)
# Periodicity.
# Joints whose positions should match after half a gait cycle.
periodicQsJointsA = ['pelvis_tilt', 'pelvis_ty', 
                     'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
                     'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 
                     'knee_angle_l', 'knee_angle_r', 
                     'ankle_angle_l', 'ankle_angle_r', 
                     'subtalar_angle_l', 'subtalar_angle_r', 
                     'mtp_angle_l', 'mtp_angle_r',
                     'lumbar_extension', 
                     'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
                     'arm_flex_r', 'arm_add_r', 'arm_rot_r', 
                     'elbow_flex_l', 'elbow_flex_r']
periodicQsJointsB = ['pelvis_tilt', 'pelvis_ty', 
                     'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 
                     'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
                     'knee_angle_r', 'knee_angle_l', 
                     'ankle_angle_r', 'ankle_angle_l', 
                     'subtalar_angle_r', 'subtalar_angle_l', 
                     'mtp_angle_r', 'mtp_angle_l',
                     'lumbar_extension', 
                     'arm_flex_r', 'arm_add_r', 'arm_rot_r', 
                     'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
                     'elbow_flex_r', 'elbow_flex_l']
# Joints whose positions and velocities should be equal and opposite after 
# half a gait cycle.
periodicOppositeJoints = ['pelvis_list', 'pelvis_rotation', 'pelvis_tz', 
                          'lumbar_bending', 'lumbar_rotation']
# Lower leg joints for which ID might not be valid.
noFPJoints = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
              'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 
              'knee_angle_l', 'knee_angle_r', 
              'ankle_angle_l', 'ankle_angle_r', 
              'subtalar_angle_l', 'subtalar_angle_r', 
              'mtp_angle_l', 'mtp_angle_r']
# EMG channels.
channels_r = ['HamL_r', 'TA_r', 'PerL_r', 'GL_r', 'HamM_r', 'Sol_r', 'VL_r',
              'VM_r', 'GluMed_r', 'RF_r']
channels_l = ['HamL_l', 'TA_l', 'PerL_l', 'GL_l', 'HamM_l', 'Sol_l', 'VL_l', 
              'VM_l', 'GluMed_l', 'RF_l', 'PerB_l', 'GM_l', 'AddL_l', 'TFL_l']
channels = channels_l + channels_r
channels_r_all = [channel_l[:-1] + 'r' for channel_l in channels_l]
channels_lr = channels_l + channels_r_all
channels_rl = channels_r_all + channels_l  
# Sides.
sides = ["right", "left"]
# Paths
pathMain = os.getcwd()
pathGRF = os.path.join(pathMain, 'OpenSimModel', "GRF")
pathEMG = os.path.join(pathMain, 'OpenSimModel', "EMG")
# Threshold for detecting contact.
threshold = 30
# Number of interpolating points.
N = 100

# Loop over subjects
GRF = {} 
kinematics = {} 
kinetics = {}
EMG = {} 
experimentalData = {}
for subject in trials:    
    pathData = os.path.join(pathMain, 'OpenSimModel', subject)
    GRF[subject] = {}
    kinematics[subject] = {}
    kinetics[subject] = {}
    EMG[subject] = {}
    experimentalData[subject] = {}    
    GRF_all = np.zeros((N, nGRFS + 1, len(trials[subject]["names"])))
    kinetics_all = np.zeros((N, nJoints + 1, len(trials[subject]["names"])))
    kinematics_all = np.zeros((N, nJoints + 1, len(trials[subject]["names"])))
    EMG_all = np.zeros((N, len(channels_rl) + 1, len(trials[subject]["names"])))    
    # Loop over trials
    for idxTrial, trial in enumerate(trials[subject]["names"]):
        
        # %% GRF
        GRF[subject][trial] = {}        
        pathGRF_trial = os.path.join(pathGRF, "GRF_" + trial + ".mot")
        # Pre-allocation                  
        IC_GRF1 = {}
        IC_GRF1["idx"] = {}
        IC_GRF1["time"] = {}        
        for side in sides:
            # Get GFRs
            GRF[subject][trial][side] = getGRF(pathGRF_trial, headers["GRF"][side])  
            # Identify initial contacts
            _, _, IC_GRF1["idx"][side], IC_GRF1["time"][side] = getInitialContact(
                GRF[subject][trial][side][headers["GRF"][side][1]].to_numpy(), 
                GRF[subject][trial][side]["time"].to_numpy(), threshold)
        # Identify which leg hits the first force plate
        if IC_GRF1["idx"]["right"] < IC_GRF1["idx"]["left"]:
            legIC = "right"   
        elif IC_GRF1["idx"]["left"] < IC_GRF1["idx"]["right"]:
            legIC = "left"   
        else:
            raise ValueError('No initial contact identified')
        # Identify index second initial contact
        IC_GRF2 = {}
        temp = (np.argwhere(GRF[subject][trial][legIC]["time"].to_numpy() <= trials[subject]["timeIC2"][idxTrial])[-1])[0]
        IC_GRF2["time"] = np.round(GRF[subject][trial][legIC]["time"].iloc[temp], 2)        
        IC_GRF2["idx"] = (np.argwhere(GRF[subject][trial][legIC]["time"].to_numpy() >= IC_GRF2["time"])[0])[0]         
        # Interpolate data
        GRF[subject][trial]["interp"] = {}
        GRF[subject][trial]["interp"]["raw"] = {}
        GRF[subject][trial]["interp"]["raw"]["right"] = interpolateDataFrame(GRF[subject][trial]["right"], IC_GRF1["time"][legIC][0], IC_GRF2["time"], N)
        GRF[subject][trial]["interp"]["raw"]["left"] = interpolateDataFrame(GRF[subject][trial]["left"], IC_GRF1["time"][legIC][0], IC_GRF2["time"], N)        
        GRF[subject][trial]["interp"]["adjusted"] = {}
        GRF[subject][trial]["interp"]["adjusted"]["right"] = GRF[subject][trial]["interp"]["raw"]["right"].copy(deep=True)
        GRF[subject][trial]["interp"]["adjusted"]["left"] = GRF[subject][trial]["interp"]["raw"]["left"].copy(deep=True)        
        # Adjust data so that gait cycle starts at right heel strike
        if legIC == "left":            
            for idxHeader, header in enumerate(headers["GRF"]["right"]):          
                if header == "1_ground_force_vz":
                    GRF[subject][trial]["interp"]["adjusted"]["right"][header] = -GRF[subject][trial]["interp"]["raw"]["left"][headers["GRF"]["left"][idxHeader]]    
                    GRF[subject][trial]["interp"]["adjusted"]["left"][headers["GRF"]["left"][idxHeader]] = -GRF[subject][trial]["interp"]["raw"]["right"][header]    
                else:
                    GRF[subject][trial]["interp"]["adjusted"]["right"][header] = GRF[subject][trial]["interp"]["raw"]["left"][headers["GRF"]["left"][idxHeader]]
                    GRF[subject][trial]["interp"]["adjusted"]["left"][headers["GRF"]["left"][idxHeader]] = GRF[subject][trial]["interp"]["raw"]["right"][header]     
        GRF[subject][trial]["interp"]["adjusted"]["all"] = GRF[subject][trial]["interp"]["adjusted"]["right"].copy(deep=True)
        for count, header in enumerate(headers["GRF"]["left"]):
            GRF[subject][trial]["interp"]["adjusted"]["all"].insert(GRF[subject][trial]["interp"]["adjusted"]["right"].shape[1] + count, header, GRF[subject][trial]["interp"]["adjusted"]["left"][header])           
        GRF[subject][trial]["interp"]["adjusted"]["all"]["time"] -= GRF[subject][trial]["interp"]["adjusted"]["all"]["time"][0]        
        GRF_all[:, :, idxTrial] = GRF[subject][trial]["interp"]["adjusted"]["all"].to_numpy()        
        for count, header in enumerate(headers["GRF"]["all"]):
            GRF[subject][trial]["interp"]["adjusted"]["all"] = GRF[subject][trial]["interp"]["adjusted"]["all"].rename(columns={header: headers["GRF_adj"]["all"][count]})
    
        # %% Kinematics
        kinematics[subject][trial] = {}
        kinematics[subject][trial]["positions"] = {}        
        pathIK_trial = os.path.join(pathData, "IK", "IK_" + trial + ".mot")        
        kinematics[subject][trial]["positions"]["all"] = getIK(pathIK_trial, joints, degrees=True)[0]
        IC_IK1 = {}
        IC_IK1["idx"] = (np.argwhere(kinematics[subject][trial]["positions"]["all"]["time"].to_numpy() >= IC_GRF1["time"][legIC][0]))[0][0]
        IC_IK1["time"] = kinematics[subject][trial]["positions"]["all"]["time"].iloc[IC_IK1["idx"]]
        IC_IK2 = {}
        IC_IK2["idx"] = (np.argwhere(kinematics[subject][trial]["positions"]["all"]["time"].to_numpy() >= IC_GRF2["time"]))[0][0]
        IC_IK2["time"] = kinematics[subject][trial]["positions"]["all"]["time"].iloc[IC_IK2["idx"]]
        kinematics[subject][trial]["positions"]["interp"] = {}
        kinematics[subject][trial]["positions"]["interp"]["raw"] = interpolateDataFrame(kinematics[subject][trial]["positions"]["all"], IC_IK1["time"], IC_IK2["time"], N)
        kinematics[subject][trial]["positions"]["interp"]["adjusted"] = {}
        kinematics[subject][trial]["positions"]["interp"]["adjusted"] = kinematics[subject][trial]["positions"]["interp"]["raw"].copy(deep=True)      
        # Adjust data so that gait cycle starts at right heel strike
        if legIC == "left":  
            for idxperiodicQsJointA, periodicQsJointA in enumerate(periodicQsJointsA):
                kinematics[subject][trial]["positions"]["interp"]["adjusted"][periodicQsJointA] = kinematics[subject][trial]["positions"]["interp"]["raw"][periodicQsJointsB[idxperiodicQsJointA]]        
            for periodicOppositeJoint in periodicOppositeJoints:
                kinematics[subject][trial]["positions"]["interp"]["adjusted"][periodicOppositeJoint] = -kinematics[subject][trial]["positions"]["interp"]["raw"][periodicOppositeJoint]            
        kinematics[subject][trial]["positions"]["interp"]["adjusted"]["pelvis_tx"] -= kinematics[subject][trial]["positions"]["interp"]["raw"]["pelvis_tx"][0]
        kinematics[subject][trial]["positions"]["interp"]["adjusted"]["time"] -= kinematics[subject][trial]["positions"]["interp"]["raw"]["time"][0]        
        kinematics_all[:, :, idxTrial] = kinematics[subject][trial]["positions"]["interp"]["adjusted"].to_numpy()
        
        # %% Kinetics
        kinetics[subject][trial] = {}        
        pathID_trial = os.path.join(pathData, "ID", "ID_" + trial + ".sto") 
        kinetics[subject][trial]["all"] = getID(pathID_trial, joints)        
        IC_ID1 = {}
        IC_ID1["idx"] = (np.argwhere(kinetics[subject][trial]["all"]["time"].to_numpy() >= IC_GRF1["time"][legIC][0]))[0][0]
        IC_ID1["time"] = kinetics[subject][trial]["all"]["time"].iloc[IC_ID1["idx"]]        
        IC_ID2 = {}
        IC_ID2["idx"] = (np.argwhere(kinetics[subject][trial]["all"]["time"].to_numpy() >= IC_GRF2["time"]))[0][0]
        IC_ID2["time"] = kinetics[subject][trial]["all"]["time"].iloc[IC_ID2["idx"]]       
        kinetics[subject][trial]["interp"] = {}
        kinetics[subject][trial]["interp"]["raw"] = interpolateDataFrame(kinetics[subject][trial]["all"], IC_ID1["time"], IC_ID2["time"], N)        
        kinetics[subject][trial]["interp"]["raw_temp"] = {}
        kinetics[subject][trial]["interp"]["raw_temp"] = kinetics[subject][trial]["interp"]["raw"].copy(deep=True)
        
        # Inverse dynamics is not valid for the leg that does not start at heel
        # strike, since that leg might be in contact with the ground but with
        # no force plate data. At heel strike, there is double support but the
        # leg in late stance is typically not on a force plate.
        # Let's replace the lower leg torques with NaNs in such cases        
        for noFPJoint in noFPJoints:
            # If heel strike on left side, then right side torques not valid.
            if legIC == "left": 
                if noFPJoint[-2:] == "_r":
                    kinetics[subject][trial]["interp"]["raw_temp"][noFPJoint] = np.NaN
            # If heel strike on right side, then left side torques not valid.
            elif legIC == "right":
                if noFPJoint[-2:] == "_l":
                    kinetics[subject][trial]["interp"]["raw_temp"][noFPJoint] = np.NaN                    
        kinetics[subject][trial]["interp"]["adjusted"] = {}
        kinetics[subject][trial]["interp"]["adjusted"] = kinetics[subject][trial]["interp"]["raw_temp"].copy(deep=True)        
        # Adjust data so that gait cycle starts at right heel strike
        if legIC == "left":  
            for idxperiodicQsJointA, periodicQsJointA in enumerate(periodicQsJointsA):
                kinetics[subject][trial]["interp"]["adjusted"][periodicQsJointA] = kinetics[subject][trial]["interp"]["raw_temp"][periodicQsJointsB[idxperiodicQsJointA]]
            for periodicOppositeJoint in periodicOppositeJoints:
                kinetics[subject][trial]["interp"]["adjusted"][periodicOppositeJoint] = -kinetics[subject][trial]["interp"]["raw_temp"][periodicOppositeJoint]                                             
        kinetics[subject][trial]["interp"]["adjusted"]["time"] -= kinetics[subject][trial]["interp"]["raw"]["time"][0]
        kinetics_all[:, :, idxTrial] = kinetics[subject][trial]["interp"]["adjusted"].to_numpy()
        
        # %% EMG
        EMG[subject][trial] = {}        
        pathGRF_trial = os.path.join(pathEMG, "EMG_" + trial + ".mot")
        EMG[subject][trial]["all"] = getFromStorage(pathGRF_trial, channels)        
        IC_EMG1 = {}
        IC_EMG1["idx"] = (np.argwhere(EMG[subject][trial]["all"]["time"].to_numpy() >= IC_GRF1["time"][legIC][0]))[0][0]
        IC_EMG1["time"] = EMG[subject][trial]["all"]["time"].iloc[IC_EMG1["idx"]]        
        IC_EMG2 = {}
        IC_EMG2["idx"] = (np.argwhere(EMG[subject][trial]["all"]["time"].to_numpy() >= IC_GRF2["time"]))[0][0]
        IC_EMG2["time"] = EMG[subject][trial]["all"]["time"].iloc[IC_EMG2["idx"]]        
        EMG[subject][trial]["interp"] = {}
        EMG[subject][trial]["interp"]["raw"] = interpolateDataFrame(EMG[subject][trial]["all"], IC_EMG1["time"], IC_EMG2["time"], N)
        
        # There are more EMG channels for the left leg than for the right leg.
        # To make things simpler, I set NaN to non-existing channels of the
        # right leg that exist for the left leg.
        for channel_l in channels_l:
            channel_r = channel_l[:-1] + "r"
            if not channel_r in EMG[subject][trial]["interp"]["raw"]:
                EMG[subject][trial]["interp"]["raw"].insert(EMG[subject][trial]["interp"]["raw"].shape[1], channel_r, np.NaN)       
        # Data is bad in certain cases
        if (trial == "gait_63" or trial == "gait_65"):
            for channel_r in channels_r:
                if not (channel_r == "Sol_r" or channel_r == "VL_r"):
                    EMG[subject][trial]["interp"]["raw"][channel_r] = np.NaN                    
        if (trial == "gait_15" or trial == "gait_27"):
            for channel_r in channels_r:
                if not (channel_r == "PerL_r" or channel_r == "HamM_r" or 
                        channel_r == "Sol_r" or channel_r == "VL_r"):
                    EMG[subject][trial]["interp"]["raw"][channel_r] = np.NaN            
        if (trial == "gait_61"):
            for channel in channels:
                if (channel == "HamL_r" or channel == "TA_r" or 
                    channel == "PerL_r" or channel == "GL_r" or
                    channel == "HamM_r" or channel == "VM_r" ):
                    EMG[subject][trial]["interp"]["raw"][channel] = np.NaN  
        if (trial == "gait_64"):
            for channel in channels:
                if (channel == "HamL_r" or channel == "TA_r" or 
                    channel == "PerL_r" or channel == "GL_r" or
                    channel == "HamM_r" or channel == "VM_r"):
                    EMG[subject][trial]["interp"]["raw"][channel] = np.NaN    
        if (trial == "gait_14"):
            for channel in channels:
                if (channel == "HamL_r" or channel == "TA_r"  or 
                    channel == "GL_r" or channel == "VM_r"):
                    EMG[subject][trial]["interp"]["raw"][channel] = np.NaN                    
        if (trial == "gait_23"):
            for channel in channels:
                if (channel == "HamL_r" or channel == "TA_r"  or 
                    channel == "GL_r" or channel == "VM_r"):
                    EMG[subject][trial]["interp"]["raw"][channel] = np.NaN                    
        if (trial == "gait_25"):
            for channel in channels:
                if (channel == "HamL_r" or channel == "TA_r"  or 
                    channel == "GL_r" or channel == "VM_r"):
                    EMG[subject][trial]["interp"]["raw"][channel] = np.NaN                    
        if (trial == "gait_60"):
            for channel in channels:
                if (channel == "HamL_r" or channel == "TA_r" or 
                    channel == "PerL_r"  or channel == "GL_r" or
                    channel == "HamM_r" or channel == "VM_r"):
                    EMG[subject][trial]["interp"]["raw"][channel] = np.NaN       
        EMG[subject][trial]["interp"]["adjusted"] = {}
        EMG[subject][trial]["interp"]["adjusted"] = EMG[subject][trial]["interp"]["raw"].copy(deep=True) 
        # Adjust data so that gait cycle starts at right heel strike
        if legIC == "left": 
            for idxChannel_lr, channel_lr in enumerate(channels_lr):
                EMG[subject][trial]["interp"]["adjusted"][channel_lr] = EMG[subject][trial]["interp"]["raw"][channels_rl[idxChannel_lr]]
        EMG[subject][trial]["interp"]["adjusted"]["time"] -= EMG[subject][trial]["interp"]["raw"]["time"][0]            
        EMG_all[:, :, idxTrial] = EMG[subject][trial]["interp"]["adjusted"].to_numpy()
        
    # %% Mean and standard deviations
    GC_percent = np.linspace(1, 100, N)
    # GRF
    GRF_mean = np.mean(GRF_all, axis=2) 
    GRF_std = np.std(GRF_all, axis=2)     
    experimentalData[subject]["GRF"] = {}
    experimentalData[subject]["GRF"]["mean"] = pd.DataFrame(data=GRF_mean, columns=GRF[subject][trial]["interp"]["adjusted"]["all"].columns.values)
    experimentalData[subject]["GRF"]["std"] = pd.DataFrame(data=GRF_std, columns=GRF[subject][trial]["interp"]["adjusted"]["all"].columns.values)
    experimentalData[subject]["GRF"]["GC_percent"] = GC_percent
    # Kinematics
    kinematics_positions_mean = np.mean(kinematics_all, axis=2)
    kinematics_positions_std = np.std(kinematics_all, axis=2)  
    experimentalData[subject]["kinematics"] = {}
    experimentalData[subject]["kinematics"]["positions"] = {}
    experimentalData[subject]["kinematics"]["positions"]["mean"] = pd.DataFrame(data=kinematics_positions_mean, columns=kinematics[subject][trial]["positions"]["interp"]["adjusted"].columns.values)
    experimentalData[subject]["kinematics"]["positions"]["std"] = pd.DataFrame(data=kinematics_positions_std, columns=kinematics[subject][trial]["positions"]["interp"]["adjusted"].columns.values)
    experimentalData[subject]["kinematics"]["positions"]["GC_percent"] = GC_percent
    # Kinetics
    kinetics_mean = np.mean(kinetics_all, axis=2)
    kinetics_std = np.std(kinetics_all, axis=2)  
    experimentalData[subject]["kinetics"] = {}
    experimentalData[subject]["kinetics"]["mean"] = pd.DataFrame(data=kinetics_mean, columns=kinetics[subject][trial]["interp"]["adjusted"].columns.values)
    experimentalData[subject]["kinetics"]["std"] = pd.DataFrame(data=kinetics_std, columns=kinetics[subject][trial]["interp"]["adjusted"].columns.values)
    experimentalData[subject]["kinetics"]["GC_percent"] = GC_percent
    # EMG
    EMG_mean = np.nanmean(EMG_all, axis=2)
    EMG_std = np.nanstd(EMG_all, axis=2)  
    experimentalData[subject]["EMG"] = {}
    experimentalData[subject]["EMG"]["mean"] = pd.DataFrame(data=EMG_mean, columns=EMG[subject][trial]["interp"]["adjusted"].columns.values)
    experimentalData[subject]["EMG"]["std"] = pd.DataFrame(data=EMG_std, columns=EMG[subject][trial]["interp"]["adjusted"].columns.values)
    experimentalData[subject]["EMG"]["GC_percent"] = GC_percent
    
    if saveExperimentalData:
        np.save(os.path.join(pathData, 'experimentalData.npy'), experimentalData)
    
# %% Plots
if plotData:
    
    # GRF
    fig, axs = plt.subplots(2, 3, sharex=True)    
    fig.suptitle('Ground reaction forces')
    GC_percent = np.linspace(1, 100, N)
    for i, ax in enumerate(axs.flat):
        color=iter(plt.cm.rainbow(np.linspace(0,1,len(trials[subject]["names"]))))   
        for idxTrial, trial in enumerate(trials[subject]["names"]):
            ax.plot(GC_percent,
                    GRF[subject][trial]["interp"]["adjusted"]["all"][headers["GRF_adj"]["all"][i]], c=next(color), label='case_' + trial)          
            
            ax.fill_between(GC_percent,
                            experimentalData[subject]["GRF"]["mean"][headers["GRF_adj"]["all"][i]] + 2*experimentalData[subject]["GRF"]["std"][headers["GRF_adj"]["all"][i]],
                            experimentalData[subject]["GRF"]["mean"][headers["GRF_adj"]["all"][i]] - 2*experimentalData[subject]["GRF"]["std"][headers["GRF_adj"]["all"][i]])            
        ax.set_title(headers["GRF_adj"]["all"][i])
    plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
    plt.setp(axs[:, 0], ylabel='(N)')
    fig.align_ylabels()
    
    # Kinematics
    fig, axs = plt.subplots(4, 6, sharex=True)    
    fig.suptitle('Joint kinematics')
    GC_percent = np.linspace(1, 100, N)
    for i, ax in enumerate(axs.flat):
        if i < len(joints):
            color=iter(plt.cm.rainbow(np.linspace(0,1,len(trials[subject]["names"]))))   
            for idxTrial, trial in enumerate(trials[subject]["names"]):
                ax.plot(GC_percent,
                        kinematics[subject][trial]["positions"]["interp"]["adjusted"][joints[i]], c=next(color), label='case_' + trial)   
                
                ax.fill_between(GC_percent,
                            experimentalData[subject]["kinematics"]["positions"]["mean"][joints[i]] + 2*experimentalData[subject]["kinematics"]["positions"]["std"][joints[i]],
                            experimentalData[subject]["kinematics"]["positions"]["mean"][joints[i]] - 2*experimentalData[subject]["kinematics"]["positions"]["std"][joints[i]])                        
            ax.set_title(joints[i])
    plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
    plt.setp(axs[:, 0], ylabel='(rad or m)')
    fig.align_ylabels()
    
    # Kinetics
    fig, axs = plt.subplots(4, 6, sharex=True)    
    fig.suptitle('Joint kinetics')
    GC_percent = np.linspace(1, 100, N)
    for i, ax in enumerate(axs.flat):
        if i < len(joints):
            color=iter(plt.cm.rainbow(np.linspace(0,1,len(trials[subject]["names"]))))   
            for idxTrial, trial in enumerate(trials[subject]["names"]):
                if not kinetics[subject][trial]["interp"]["adjusted"][joints[i]][0] == np.NaN:
                    ax.plot(GC_percent,
                            kinetics[subject][trial]["interp"]["adjusted"][joints[i]], c=next(color), label='case_' + trial)    
                    
                    ax.fill_between(GC_percent,
                            experimentalData[subject]["kinetics"]["mean"][joints[i]] + 2*experimentalData[subject]["kinetics"]["std"][joints[i]],
                            experimentalData[subject]["kinetics"]["mean"][joints[i]] - 2*experimentalData[subject]["kinetics"]["std"][joints[i]])                    
            ax.set_title(joints[i])
    plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
    plt.setp(axs[:, 0], ylabel='[Nm]')
    fig.align_ylabels()   

    # EMG
    fig, axs = plt.subplots(4, 7, sharex=True)    
    fig.suptitle('EMG')
    GC_percent = np.linspace(1, 100, N)
    for i, ax in enumerate(axs.flat):
        if i < len(channels):
            color=iter(plt.cm.rainbow(np.linspace(0,1,len(trials[subject]["names"]))))   
            for idxTrial, trial in enumerate(trials[subject]["names"]):
                if not EMG[subject][trial]["interp"]["adjusted"][channels[i]][0] == np.NaN:
                    ax.plot(GC_percent,
                            EMG[subject][trial]["interp"]["adjusted"][channels[i]], c=next(color), label='case_' + trial)                        
                    ax.fill_between(GC_percent,
                            experimentalData[subject]["EMG"]["mean"][channels[i]] + 2*experimentalData[subject]["EMG"]["std"][channels[i]],
                            experimentalData[subject]["EMG"]["mean"][channels[i]] - 2*experimentalData[subject]["EMG"]["std"][channels[i]])                    
            ax.set_title(channels[i])
    plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
    plt.setp(axs[:, 0], ylabel='[-]')
    fig.align_ylabels()         
        