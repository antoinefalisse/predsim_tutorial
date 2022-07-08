'''
    This script contains classes to set bounds to the optimization variables.
'''

# %% Import packages
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np

# %% Class bounds.
class bounds:
    
    def __init__(self, Qs, joints, muscles, armJoints, targetSpeed, 
                 mtpJoints=['mtp_angle_l', 'mtp_angle_r']):
        
        self.Qs = Qs
        self.joints = joints
        self.targetSpeed = targetSpeed
        self.muscles = muscles
        self.armJoints = armJoints
        self.mtpJoints = mtpJoints
        
    def splineQs(self):        
        self.Qs_spline = self.Qs.copy()
        self.Qdots_spline = self.Qs.copy()
        self.Qdotdots_spline = self.Qs.copy()

        for joint in self.joints:
            spline = interpolate.InterpolatedUnivariateSpline(self.Qs['time'], 
                                                              self.Qs[joint],
                                                              k=3)
            self.Qs_spline[joint] = spline(self.Qs['time'])
            splineD1 = spline.derivative(n=1)
            self.Qdots_spline[joint] = splineD1(self.Qs['time'])
            splineD2 = spline.derivative(n=2)
            self.Qdotdots_spline[joint] = splineD2(self.Qs['time'])
    
    def getBoundsPosition(self):        
        self.splineQs()
        upperBoundsPosition = pd.DataFrame()   
        lowerBoundsPosition = pd.DataFrame() 
        scalingPosition = pd.DataFrame() 
        for count, joint in enumerate(self.joints):  
            if (joint == 'mtp_angle_l') or (joint == 'mtp_angle_r'):
                upperBoundsPosition.insert(count, joint, [1.05])
                lowerBoundsPosition.insert(count, joint, [-0.5])                 
            else:              
                if (self.joints.count(joint[:-1] + 'l')) == 1:        
                    ub = max(max(self.Qs_spline[joint[:-1] + 'l']), 
                             max(self.Qs_spline[joint[:-1] + 'r']))
                    lb = min(min(self.Qs_spline[joint[:-1] + 'l']), 
                             min(self.Qs_spline[joint[:-1] + 'r']))                              
                else:
                    ub = max(self.Qs_spline[joint])
                    lb = min(self.Qs_spline[joint])
                r = abs(ub - lb)
                ub = ub + 2*r
                lb = lb - 2*r                        
                upperBoundsPosition.insert(count, joint, [ub])
                lowerBoundsPosition.insert(count, joint, [lb]) 
                
                # Special cases
                if joint == 'pelvis_tx':
                    upperBoundsPosition[joint] = [2]
                    lowerBoundsPosition[joint] = [0]
                elif joint == 'pelvis_ty':
                    upperBoundsPosition[joint] = [1.1]
                    lowerBoundsPosition[joint] = [0.75]
                elif joint == 'pelvis_tz':
                    upperBoundsPosition[joint] = [0.1]
                    lowerBoundsPosition[joint] = [-0.1]
                elif (joint == 'elbow_flex_l') or (joint == 'elbow_flex_r'):
                    lowerBoundsPosition[joint] = [0]
                elif ((joint == 'arm_add_l') or (joint == 'arm_rot_l') or 
                      (joint == 'arm_add_r') or (joint == 'arm_rot_r')):
                    ub = max(max(self.Qs_spline[joint[:-1] + 'l']), 
                             max(self.Qs_spline[joint[:-1] + 'r']))
                    upperBoundsPosition[joint] = [ub]
                elif joint == 'pelvis_tilt':
                    lowerBoundsPosition[joint] = [-20*np.pi/180]
                if self.targetSpeed > 1.33:
                    if joint == 'arm_flex_r':
                        lowerBoundsPosition[joint] = [-50*np.pi/180]
                    if joint == 'arm_flex_l':
                        lowerBoundsPosition[joint] = [-50*np.pi/180]
                
            # Scaling.          
            s = np.max(np.array([abs(upperBoundsPosition[joint])[0],
                                 abs(lowerBoundsPosition[joint])[0]]))
            scalingPosition.insert(count, joint, [s])
            lowerBoundsPosition[joint] /= scalingPosition[joint]
            upperBoundsPosition[joint] /= scalingPosition[joint]
            
        # Hard bounds at initial position.
        lowerBoundsPositionInitial = lowerBoundsPosition.copy()
        lowerBoundsPositionInitial['pelvis_tx'] = [0]
        upperBoundsPositionInitial = upperBoundsPosition.copy()
        upperBoundsPositionInitial['pelvis_tx'] = [0]
                
        return (upperBoundsPosition, lowerBoundsPosition, scalingPosition,
                upperBoundsPositionInitial, lowerBoundsPositionInitial) 
    
    def getBoundsVelocity(self):        
        self.splineQs()
        upperBoundsVelocity = pd.DataFrame()   
        lowerBoundsVelocity = pd.DataFrame() 
        scalingVelocity = pd.DataFrame() 
        for count, joint in enumerate(self.joints):  
            if (joint == 'mtp_angle_l') or (joint == 'mtp_angle_r'):
                upperBoundsVelocity.insert(count, joint, [13])
                lowerBoundsVelocity.insert(count, joint, [-13]) 
            else:            
                if (self.joints.count(joint[:-1] + 'l')) == 1:        
                    ub = max(max(self.Qdots_spline[joint[:-1] + 'l']), 
                             max(self.Qdots_spline[joint[:-1] + 'r']))
                    lb = min(min(self.Qdots_spline[joint[:-1] + 'l']), 
                             min(self.Qdots_spline[joint[:-1] + 'r']))                              
                else:
                    ub = max(self.Qdots_spline[joint])
                    lb = min(self.Qdots_spline[joint])
                r = abs(ub - lb)
                ub = ub + 3*r
                lb = lb - 3*r                        
                upperBoundsVelocity.insert(count, joint, [ub])
                lowerBoundsVelocity.insert(count, joint, [lb])
    
                # Special cases.
                if self.targetSpeed > 1.33:
                    upperBoundsVelocity['pelvis_tx'] = [4]

            # Scaling.
            s = np.max(np.array([abs(upperBoundsVelocity[joint])[0],
                                 abs(lowerBoundsVelocity[joint])[0]]))            
            scalingVelocity.insert(count, joint, [s])
            upperBoundsVelocity[joint] /= scalingVelocity[joint]
            lowerBoundsVelocity[joint] /= scalingVelocity[joint]

        return upperBoundsVelocity, lowerBoundsVelocity, scalingVelocity
    
    def getBoundsAcceleration(self):        
        self.splineQs()
        upperBoundsAcceleration = pd.DataFrame()   
        lowerBoundsAcceleration = pd.DataFrame() 
        scalingAcceleration = pd.DataFrame() 
        for count, joint in enumerate(self.joints):     
            if (joint == 'mtp_angle_l') or (joint == 'mtp_angle_r'):
                upperBoundsAcceleration.insert(count, joint, [500])
                lowerBoundsAcceleration.insert(count, joint, [-500]) 
            else:           
                if (self.joints.count(joint[:-1] + 'l')) == 1:        
                    ub = max(max(self.Qdotdots_spline[joint[:-1] + 'l']), 
                             max(self.Qdotdots_spline[joint[:-1] + 'r']))
                    lb = min(min(self.Qdotdots_spline[joint[:-1] + 'l']), 
                             min(self.Qdotdots_spline[joint[:-1] + 'r']))                              
                else:
                    ub = max(self.Qdotdots_spline[joint])
                    lb = min(self.Qdotdots_spline[joint])
                r = abs(ub - lb)
                ub = ub + 3*r
                lb = lb - 3*r                        
                upperBoundsAcceleration.insert(count, joint, [ub])
                lowerBoundsAcceleration.insert(count, joint, [lb])   
            
            # Scaling.
            s = np.max(np.array([abs(upperBoundsAcceleration[joint])[0],
                                 abs(lowerBoundsAcceleration[joint])[0]]))
            scalingAcceleration.insert(count, joint, [s])
            upperBoundsAcceleration[joint] /= scalingAcceleration[joint]
            lowerBoundsAcceleration[joint] /= scalingAcceleration[joint]

        return (upperBoundsAcceleration, lowerBoundsAcceleration, 
                scalingAcceleration)
    
    def getBoundsActivation(self):
        lb = [0.05] 
        lb_vec = lb * len(self.muscles)
        ub = [1]
        ub_vec = ub * len(self.muscles)
        s = [1]
        s_vec = s * len(self.muscles)
        upperBoundsActivation = pd.DataFrame([ub_vec], columns=self.muscles)   
        lowerBoundsActivation = pd.DataFrame([lb_vec], columns=self.muscles) 
        scalingActivation = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsActivation = upperBoundsActivation.div(scalingActivation)
        lowerBoundsActivation = lowerBoundsActivation.div(scalingActivation)
        for count, muscle in enumerate(self.muscles):
            upperBoundsActivation.insert(count + len(self.muscles), 
                                         muscle[:-1] + 'l', ub)
            lowerBoundsActivation.insert(count + len(self.muscles), 
                                         muscle[:-1] + 'l', lb)  

            # Scaling.                     
            scalingActivation.insert(count + len(self.muscles), 
                                     muscle[:-1] + 'l', s)  
            upperBoundsActivation[
                    muscle[:-1] + 'l'] /= scalingActivation[muscle[:-1] + 'l']
            lowerBoundsActivation[
                    muscle[:-1] + 'l'] /= scalingActivation[muscle[:-1] + 'l']
        
        return upperBoundsActivation, lowerBoundsActivation, scalingActivation
    
    def getBoundsForce(self):
        lb = [0] 
        lb_vec = lb * len(self.muscles)
        ub = [5]
        ub_vec = ub * len(self.muscles)
        s = max([abs(lbi) for lbi in lb], [abs(ubi) for ubi in ub])
        s_vec = s * len(self.muscles)
        upperBoundsForce = pd.DataFrame([ub_vec], columns=self.muscles)   
        lowerBoundsForce = pd.DataFrame([lb_vec], columns=self.muscles) 
        scalingForce = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsForce = upperBoundsForce.div(scalingForce)
        lowerBoundsForce = lowerBoundsForce.div(scalingForce)
        for count, muscle in enumerate(self.muscles):
            upperBoundsForce.insert(count + len(self.muscles), 
                                    muscle[:-1] + 'l', ub)
            lowerBoundsForce.insert(count + len(self.muscles), 
                                    muscle[:-1] + 'l', lb)  

            # Scaling.                  
            scalingForce.insert(count + len(self.muscles), 
                                         muscle[:-1] + 'l', s)   
            upperBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
            lowerBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
        
        return upperBoundsForce, lowerBoundsForce, scalingForce
    
    def getBoundsActivationDerivative(self):
        activationTimeConstant = 0.015
        deactivationTimeConstant = 0.06
        lb = [-1 / deactivationTimeConstant] 
        lb_vec = lb * len(self.muscles)
        ub = [1 / activationTimeConstant]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsActivationDerivative = pd.DataFrame([ub_vec], 
                                                       columns=self.muscles)   
        lowerBoundsActivationDerivative = pd.DataFrame([lb_vec], 
                                                       columns=self.muscles) 
        scalingActivationDerivative = pd.DataFrame([s_vec], 
                                                   columns=self.muscles)
        upperBoundsActivationDerivative = upperBoundsActivationDerivative.div(
                scalingActivationDerivative)
        lowerBoundsActivationDerivative = lowerBoundsActivationDerivative.div(
                scalingActivationDerivative)
        for count, muscle in enumerate(self.muscles):
            upperBoundsActivationDerivative.insert(count + len(self.muscles), 
                                                   muscle[:-1] + 'l', ub)
            lowerBoundsActivationDerivative.insert(count + len(self.muscles), 
                                                   muscle[:-1] + 'l', lb) 

            # Scaling.                      
            scalingActivationDerivative.insert(count + len(self.muscles), 
                                               muscle[:-1] + 'l', s)  
            upperBoundsActivationDerivative[muscle[:-1] + 'l'] /= (
                    scalingActivationDerivative[muscle[:-1] + 'l'])
            lowerBoundsActivationDerivative[muscle[:-1] + 'l'] /= (
                    scalingActivationDerivative[muscle[:-1] + 'l'])             
        
        return (upperBoundsActivationDerivative, 
                lowerBoundsActivationDerivative, scalingActivationDerivative)
    
    def getBoundsForceDerivative(self):
        lb = [-100] 
        lb_vec = lb * len(self.muscles)
        ub = [100]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsForceDerivative = pd.DataFrame([ub_vec], 
                                                  columns=self.muscles)   
        lowerBoundsForceDerivative = pd.DataFrame([lb_vec], 
                                                  columns=self.muscles) 
        scalingForceDerivative = pd.DataFrame([s_vec], 
                                                   columns=self.muscles)
        upperBoundsForceDerivative = upperBoundsForceDerivative.div(
                scalingForceDerivative)
        lowerBoundsForceDerivative = lowerBoundsForceDerivative.div(
                scalingForceDerivative)
        for count, muscle in enumerate(self.muscles):
            upperBoundsForceDerivative.insert(count + len(self.muscles), 
                                              muscle[:-1] + 'l', ub)
            lowerBoundsForceDerivative.insert(count + len(self.muscles), 
                                              muscle[:-1] + 'l', lb)   
            
            # Scaling.                      
            scalingForceDerivative.insert(count + len(self.muscles), 
                                               muscle[:-1] + 'l', s)  
            upperBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l'])
            lowerBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l']) 
        
        return (upperBoundsForceDerivative, lowerBoundsForceDerivative, 
                scalingForceDerivative)
    
    def getBoundsArmExcitation(self):
        lb = [-1] 
        lb_vec = lb * len(self.armJoints)
        ub = [1]
        ub_vec = ub * len(self.armJoints)
        s = [150]
        s_vec = s * len(self.armJoints)
        upperBoundsArmExcitation = pd.DataFrame([ub_vec], 
                                                columns=self.armJoints)   
        lowerBoundsArmExcitation = pd.DataFrame([lb_vec], 
                                                columns=self.armJoints)            
        # Scaling.
        scalingArmExcitation = pd.DataFrame([s_vec], columns=self.armJoints)
        
        return (upperBoundsArmExcitation, lowerBoundsArmExcitation,
                scalingArmExcitation)
    
    def getBoundsArmActivation(self):
        lb = [-1] 
        lb_vec = lb * len(self.armJoints)
        ub = [1]
        ub_vec = ub * len(self.armJoints)
        s = [150]
        s_vec = s * len(self.armJoints)
        upperBoundsArmActivation = pd.DataFrame([ub_vec], 
                                                columns=self.armJoints)   
        lowerBoundsArmActivation = pd.DataFrame([lb_vec], 
                                                columns=self.armJoints) 
        # Scaling.
        scalingArmActivation = pd.DataFrame([s_vec], columns=self.armJoints)                  
        
        return (upperBoundsArmActivation, lowerBoundsArmActivation, 
                scalingArmActivation)
        
    def getBoundsMtpExcitation(self):
        lb = [-1] 
        lb_vec = lb * len(self.mtpJoints)
        ub = [1]
        ub_vec = ub * len(self.mtpJoints)
        s = [30]
        s_vec = s * len(self.mtpJoints)
        upperBoundsMtpExcitation = pd.DataFrame([ub_vec], 
                                                columns=self.mtpJoints)   
        lowerBoundsMtpExcitation = pd.DataFrame([lb_vec], 
                                                columns=self.mtpJoints)            
        # Scaling.
        scalingMtpExcitation = pd.DataFrame([s_vec], columns=self.mtpJoints)
        
        return (upperBoundsMtpExcitation, lowerBoundsMtpExcitation,
                scalingMtpExcitation)
    
    def getBoundsMtpActivation(self):
        lb = [-1] 
        lb_vec = lb * len(self.mtpJoints)
        ub = [1]
        ub_vec = ub * len(self.mtpJoints)
        s = [30]
        s_vec = s * len(self.mtpJoints)
        upperBoundsMtpActivation = pd.DataFrame([ub_vec], 
                                                columns=self.mtpJoints)   
        lowerBoundsMtpActivation = pd.DataFrame([lb_vec], 
                                                columns=self.mtpJoints) 
        # Scaling.
        scalingMtpActivation = pd.DataFrame([s_vec], columns=self.mtpJoints)                  
        
        return (upperBoundsMtpActivation, lowerBoundsMtpActivation, 
                scalingMtpActivation)
    
    def getBoundsFinalTime(self):
        upperBoundsFinalTime = pd.DataFrame([1], columns=['time'])   
        lowerBoundsFinalTime = pd.DataFrame([0.1], columns=['time'])  
        
        return upperBoundsFinalTime, lowerBoundsFinalTime
