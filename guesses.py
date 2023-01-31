'''
    This script contains classes to set initial guesses to the optimization
    variables. Two classes are implemented: one for a hot-start and one for
    a cold-start.
'''

# %% Import packages
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d

# %% This class sets a cold-start guess for the optimization variables.
# All variables are constant except for the pelvis moving forward at a
# constant speed.
class coldStart:
    
    def __init__(self, N, d, joints, muscles, targetSpeed):        
        self.N = N
        self.d = d
        self.joints = joints
        self.targetSpeed = targetSpeed
        self.muscles = muscles
        
    def getGuessFinalTime(self, gaitCycleSimulation):
        allSpeeds = np.linspace(0.73, 2.73,  21)
        if gaitCycleSimulation == 'half':
            allFinalTimes = np.linspace(0.70, 0.35, 21)
        elif gaitCycleSimulation == 'full':
            allFinalTimes = np.linspace(0.70*2, 0.35*2, 21)
        idxTargetSpeed = np.argmax(allSpeeds >= self.targetSpeed)
        self.guessFinalTime = allFinalTimes[idxTargetSpeed]
        
        return self.guessFinalTime
    
    # Mesh points.
    def getGuessPosition(self, scaling):
        g = [0] * (self.N + 1)
        g_pelvis_tx = np.linspace(0, self.guessFinalTime * self.targetSpeed, 
                                  self.N)
        g_pelvis_tx = np.append(g_pelvis_tx, g_pelvis_tx[-1] + 
                                (g_pelvis_tx[-1] - g_pelvis_tx[-2]))
        g_pelvis_ty =  [0.9385] * (self.N + 1)
        self.guessPosition = pd.DataFrame()  
        for count, joint in enumerate(self.joints): 
            if joint == 'pelvis_tx':
                self.guessPosition.insert(count, joint, 
                                          g_pelvis_tx / scaling.iloc[0][joint])
            elif joint == 'pelvis_ty':
                self.guessPosition.insert(count, joint, 
                                          g_pelvis_ty / scaling.iloc[0][joint])                    
            else:
                self.guessPosition.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
        
        return self.guessPosition
    
    def getGuessVelocity(self, scaling):
        g = [0] * (self.N + 1)
        g_pelvis_tx =  [self.targetSpeed] * (self.N + 1)
        self.guessVelocity = pd.DataFrame()  
        for count, joint in enumerate(self.joints): 
            if joint == 'pelvis_tx':
                self.guessVelocity.insert(count, joint,
                                          g_pelvis_tx / scaling.iloc[0][joint])             
            else:
                self.guessVelocity.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
        
        return self.guessVelocity
    
    def getGuessAcceleration(self, scaling):
        g = [0] * self.N
        self.guessAcceleration = pd.DataFrame()  
        for count, joint in enumerate(self.joints):
            self.guessAcceleration.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
            
        return self.guessAcceleration
    
    def getGuessActivation(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessActivation = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessActivation.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessActivation
    
    def getGuessActivationDerivative(self, scaling):
        g = [0.01] * self.N
        guessActivationDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            guessActivationDerivative.insert(count, muscle, 
                                             g / scaling.iloc[0][muscle])
            
        return guessActivationDerivative
    
    def getGuessForce(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessForce = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForce.insert(count, muscle, g / scaling.iloc[0][muscle])
            
        return self.guessForce
    
    def getGuessForceDerivative(self, scaling):
        g = [0.01] * self.N
        self.guessForceDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForceDerivative.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessForceDerivative
    
    def getGuessTorqueActuatorActivation(self, torqueActuatorJoints):
        g = [0.1] * (self.N + 1)
        self.guessTorqueActuatorActivation = pd.DataFrame()  
        for count, torqueActuatorJoint in enumerate(torqueActuatorJoints):
            self.guessTorqueActuatorActivation.insert(
                    count, torqueActuatorJoint, g)
            
        return self.guessTorqueActuatorActivation
    
    def getGuessTorqueActuatorExcitation(self, torqueActuatorJoints):
        g = [0.1] * self.N
        guessTorqueActuatorExcitation = pd.DataFrame()  
        for count, torqueActuatorJoint in enumerate(torqueActuatorJoints):
            guessTorqueActuatorExcitation.insert(count, torqueActuatorJoint, g)
            
        return guessTorqueActuatorExcitation 
    
    # Collocation points.
    def getGuessActivationCol(self):            
        guessActivationCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessActivationCol = guessActivationCol.append(
                        self.guessActivation.iloc[k], ignore_index=True)
            
        return guessActivationCol
    
    def getGuessForceCol(self):
        guessForceCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessForceCol = guessForceCol.append(
                        self.guessForce.iloc[k], ignore_index=True)
            
        return guessForceCol
    
    def getGuessForceDerivativeCol(self):
        guessForceDerivativeCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessForceDerivativeCol = guessForceDerivativeCol.append(
                        self.guessForceDerivative.iloc[k], ignore_index=True)
            
        return guessForceDerivativeCol
    
    def getGuessTorqueActuatorActivationCol(self, torqueActuatorJoints):
        guessTorqueActuatorActivationCol = (
                pd.DataFrame(columns=torqueActuatorJoints))         
        for k in range(self.N):
            for c in range(self.d):          
                guessTorqueActuatorActivationCol = (
                        guessTorqueActuatorActivationCol.append(
                        self.guessTorqueActuatorActivation.iloc[k], 
                        ignore_index=True))
            
        return guessTorqueActuatorActivationCol        
    
    def getGuessPositionCol(self):
        guessPositionCol = pd.DataFrame(columns=self.joints)          
        for k in range(self.N):
            for c in range(self.d):          
                guessPositionCol = guessPositionCol.append(
                        self.guessPosition.iloc[k], ignore_index=True)
        
        return guessPositionCol
    
    def getGuessVelocityCol(self):
        guessVelocityCol = pd.DataFrame(columns=self.joints)       
        for k in range(self.N):
            for c in range(self.d):          
                guessVelocityCol = guessVelocityCol.append(
                        self.guessVelocity.iloc[k], ignore_index=True)
        
        return guessVelocityCol
    
    def getGuessAccelerationCol(self):
        guessAccelerationCol = pd.DataFrame(columns=self.joints)  
        for k in range(self.N):
            for c in range(self.d):          
                guessAccelerationCol = guessAccelerationCol.append(
                        self.guessAcceleration.iloc[k], ignore_index=True)
                
        return guessAccelerationCol
    
# %% This class sets a hot-start guess for the optimization variables.
# Joints positions, velocities, and accelerations are based on experimental
# data. Muscle variables are set to constant values.
class hotStart:
    
    def __init__(self, Qs, N, d, joints, muscles, targetSpeed, 
                 periodicQsJointsA, periodicQdotsJointsA, 
                 periodicOppositeJoints=[]):
        
        self.Qs = Qs
        self.N = N
        self.d = d
        self.joints = joints
        self.targetSpeed = targetSpeed
        self.muscles = muscles
        self.periodicQsJointsA = periodicQsJointsA
        self.periodicOppositeJoints = periodicOppositeJoints
        self.periodicQdotsJointsA = periodicQdotsJointsA
        
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
            
    def interpQs(self):
        self.splineQs()            
        tOut = np.linspace(self.Qs['time'].iloc[0], 
                           self.Qs['time'].iloc[-1], 
                           self.N + 1)    
        
        self.Qs_spline_interp = pd.DataFrame()  
        self.Qdots_spline_interp = pd.DataFrame()  
        self.Qdotdots_spline_interp = pd.DataFrame()  
        for count, joint in enumerate(self.joints):  
            set_interp = interp1d(self.Qs['time'], self.Qs_spline[joint])
            self.Qs_spline_interp.insert(count, joint, set_interp(tOut))
            
            set_interp = interp1d(self.Qs['time'], self.Qdots_spline[joint])
            self.Qdots_spline_interp.insert(count, joint, set_interp(tOut))
            
            set_interp = interp1d(self.Qs['time'], self.Qdotdots_spline[joint])
            self.Qdotdots_spline_interp.insert(count, joint, set_interp(tOut))
        
    def getGuessFinalTime(self, gaitCycleSimulation='half'):
        allSpeeds = np.linspace(0.73, 2.73,  21)
        if gaitCycleSimulation == 'half':
            allFinalTimes = np.linspace(0.70, 0.35, 21)
        elif gaitCycleSimulation == 'full':
            allFinalTimes = np.linspace(0.70*2, 0.35*2, 21)
        idxTargetSpeed = np.argmax(allSpeeds >= self.targetSpeed)
        self.guessFinalTime = allFinalTimes[idxTargetSpeed]
        
        return self.guessFinalTime
    
    # Mesh points.
    def getGuessPosition(self, scaling, adjustInitialStatePelvis_tx=True):
        self.interpQs()
        self.guessPosition = pd.DataFrame()
        for count, joint in enumerate(self.joints):
            self.guessPosition.insert(count, joint, 
                                      self.Qs_spline_interp[joint] / 
                                      scaling.iloc[0][joint])        
        if adjustInitialStatePelvis_tx:        
            self.guessPosition['pelvis_tx'] -= (
                    self.guessPosition.iloc[0]['pelvis_tx'])
        
        return self.guessPosition
    
    def getGuessVelocity(self, scaling):
        self.interpQs()
        self.guessVelocity = pd.DataFrame()
        for count, joint in enumerate(self.joints):
            self.guessVelocity.insert(count, joint, 
                                      self.Qdots_spline_interp[joint] / 
                                      scaling.iloc[0][joint])
        
        return self.guessVelocity
    
    def getGuessAcceleration(self, scaling, nullGuessAcceleration=False):
        self.interpQs()
        self.guessAcceleration = pd.DataFrame()  
        g = [0] * self.N
        for count, joint in enumerate(self.joints):                              
            if nullGuessAcceleration:
                self.guessAcceleration.insert(count, joint,
                                              g / scaling.iloc[0][joint])
            else:
                self.guessAcceleration.insert(
                    count, joint, self.Qdotdots_spline_interp[joint] /
                    scaling.iloc[0][joint]) 
                    
        return self.guessAcceleration
    
    def getGuessActivation(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessActivation = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessActivation.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessActivation
    
    def getGuessActivationDerivative(self, scaling):
        g = [0.01] * self.N
        guessActivationDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            guessActivationDerivative.insert(count, muscle, 
                                             g / scaling.iloc[0][muscle])
            
        return guessActivationDerivative
    
    def getGuessForce(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessForce = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForce.insert(count, muscle, g / scaling.iloc[0][muscle])
            
        return self.guessForce
    
    def getGuessForceDerivative(self, scaling):
        g = [0.01] * self.N
        self.guessForceDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForceDerivative.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessForceDerivative
    
    def getGuessTorqueActuatorActivation(self, torqueActuatorJoints):
        g = [0.1] * (self.N + 1)
        self.guessTorqueActuatorActivation = pd.DataFrame()  
        for count, torqueActuatorJoint in enumerate(torqueActuatorJoints):
            self.guessTorqueActuatorActivation.insert(
                    count, torqueActuatorJoint, g)
            
        return self.guessTorqueActuatorActivation
    
    def getGuessTorqueActuatorExcitation(self, torqueActuatorJoints):
        g = [0.1] * self.N
        guessTorqueActuatorExcitation = pd.DataFrame()  
        for count, torqueActuatorJoint in enumerate(torqueActuatorJoints):
            guessTorqueActuatorExcitation.insert(
                    count, torqueActuatorJoint, g)
            
        return guessTorqueActuatorExcitation   
    
    # Collocation points.
    def getGuessActivationCol(self):            
        temp = []
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessActivation.iloc[k])
        guessActivationCol = pd.DataFrame.from_records(temp)
        
        return guessActivationCol
    
    def getGuessForceCol(self):
        temp = []
        for k in range(self.N):
            for c in range(self.d):          
                temp.append(self.guessForce.iloc[k])
        guessForceCol = pd.DataFrame.from_records(temp)
            
        return guessForceCol
    
    def getGuessForceDerivativeCol(self):
        temp = []         
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessForceDerivative.iloc[k])
        guessForceDerivativeCol = pd.DataFrame.from_records(temp)
            
        return guessForceDerivativeCol
    
    def getGuessTorqueActuatorActivationCol(self, torqueActuatorJoints):
        temp = []
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessTorqueActuatorActivation.iloc[k])
        guessTorqueActuatorActivationCol = pd.DataFrame.from_records(temp)
            
        return guessTorqueActuatorActivationCol  
    
    def getGuessPositionCol(self):
        temp = []
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessPosition.iloc[k])
        guessPositionCol = pd.DataFrame.from_records(temp)
        
        return guessPositionCol
    
    def getGuessVelocityCol(self):
        temp = []      
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessVelocity.iloc[k])
        guessVelocityCol = pd.DataFrame.from_records(temp)
        
        return guessVelocityCol
    
    def getGuessAccelerationCol(self):
        temp = []
        guessAccelerationCol = pd.DataFrame(columns=self.joints)  
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessAcceleration.iloc[k])
        guessAccelerationCol = pd.DataFrame.from_records(temp)
                
        return guessAccelerationCol    
