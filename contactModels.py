'''
    This script contains classes that implement contact models.
'''

# %% Import packages.
import numpy as np
import casadi as ca

# %% This class implements a smooth approximation of Simbody's Hunt-Crossley
# contact model: https://github.com/simbody/simbody/blob/master/Simbody/include/simbody/internal/SmoothSphereHalfSpaceForce.h
class smoothSphereHalfSpaceForce:
    
    def __init__(self, stiffness, radius, dissipation, transitionVelocity,
                 staticFriction, dynamicFriction, viscousFriction, normal):
        
        self.stiffness = stiffness
        self.radius = radius
        self.dissipation = dissipation
        self.transitionVelocity = transitionVelocity
        self.staticFriction = staticFriction
        self.dynamicFriction = dynamicFriction
        self.viscousFriction = viscousFriction
        self.normal = normal
        
    def getContactForce(self, locSphere_inB, posB_inG, lVelB_inG, aVelB_inG,
                        RBG_inG, TBG_inG):
        
        # Constant values.
        eps = 1e-5
        eps2 = 1e-16
        bv = 50
        bd = 300      
        # Express sphere position in ground.
        locSphere_inG = (np.matmul(RBG_inG, locSphere_inB)+TBG_inG).T
        # Contact point position.
        locContact_inG = locSphere_inG - np.array([[0, self.radius, 0]])
        # Indentation.
        indentation = -locContact_inG[0, 1]
        # Contact point velocity.       
        term1 = aVelB_inG.T
        term2 = (locSphere_inG - np.array([[0, self.radius, 0]]) +
                0.5 * np.array([[0, indentation, 0]]) - posB_inG.T)    
        crossProduct0 = term1[0, 1]*term2[0, 2] - term1[0, 2]*term2[0, 1]
        crossProduct1 = term1[0, 2]*term2[0, 0] - term1[0, 0]*term2[0, 2]
        crossProduct2 = term1[0, 0]*term2[0, 1] - term1[0, 1]*term2[0, 0]    
        v = lVelB_inG.T + [[crossProduct0], [crossProduct1], [crossProduct2]]
        vnormal = (v[0, 0]*self.normal[0, 0] + v[0, 1]*self.normal[0, 1] + 
                   v[0, 2]*self.normal[0, 2])
        vtangent =  v - vnormal * self.normal
        indentationVel = -vnormal
        # Stiffness force.
        k = 0.5 * (self.stiffness) ** (2/3)
        fH = ((4/3) * k * np.sqrt(self.radius * k) * 
              ((np.sqrt(indentation * indentation + eps)) ** (3/2)))
        # Dissipation force.
        fHd = fH * (1 + 1.5 * self.dissipation * indentationVel)
        fn = ((0.5 * 
               np.tanh(bv * (indentationVel + 1 / (1.5 * self.dissipation))) + 
               0.5 + eps2) * (0.5 * np.tanh(bd * indentation) + 0.5 + eps2) *
              fHd)
        force = fn * self.normal
        # Friction force.
        aux = ((vtangent[0, 0]) ** 2 + (vtangent[0, 1]) ** 2 +
               (vtangent[0, 2]) ** 2 + eps)
        vslip = aux ** (0.5)
        vrel = vslip / self.transitionVelocity
        ffriction = fn * (np.fmin(vrel, 1) * (self.dynamicFriction + 
                          2 * (self.staticFriction - self.dynamicFriction) / 
                          (1 + vrel * vrel)) + self.viscousFriction * vslip)
        # Contact force.
        contactForce = force + ffriction * (-vtangent) / vslip
        
        return contactForce

# %% CasADi-specific implementation.  
class smoothSphereHalfSpaceForce_ca:
    
    def __init__(self, stiffness, radius, dissipation, transitionVelocity,
                 staticFriction, dynamicFriction, viscousFriction, normal):
        
        self.stiffness = stiffness
        self.radius = radius
        self.dissipation = dissipation
        self.transitionVelocity = transitionVelocity
        self.staticFriction = staticFriction
        self.dynamicFriction = dynamicFriction
        self.viscousFriction = viscousFriction
        self.normal = normal
        
    def getContactForce(self, locSphere_inB, posB_inG, lVelB_inG, aVelB_inG,
                        RBG_inG, TBG_inG):
        
        # Constant values.
        eps = 1e-5
        eps2 = 1e-16
        bv = 50
        bd = 300      
        # Express sphere position in ground.
        locSphere_inG = (np.matmul(RBG_inG, locSphere_inB)+TBG_inG).T
        # Contact point position.
        locContact_inG = locSphere_inG - np.array([[0, self.radius, 0]])
        # Indentation.
        indentation = -locContact_inG[0, 1]
        # Contact point velocity.
        v = lVelB_inG.T +  ca.cross(aVelB_inG.T, 
                                    locSphere_inG - 
                                    np.array([[0, self.radius, 0]]) + 
                                    0.5 * np.array([[0, indentation, 0]]) - 
                                    posB_inG.T)
        vnormal = (v[0, 0]*self.normal[0, 0] + v[0, 1]*self.normal[0, 1] +
                   v[0, 2]*self.normal[0, 2])
        vtangent =  v - vnormal * self.normal
        indentationVel = -vnormal
        # Stiffness force.
        k = 0.5 * (self.stiffness) ** (2/3)
        fH = ((4/3) * k * np.sqrt(self.radius * k) * 
              ((np.sqrt(indentation * indentation + eps)) ** (3/2)))
        # Dissipation force.
        fHd = fH * (1 + 1.5 * self.dissipation * indentationVel)
        fn = ((0.5 * 
               np.tanh(bv * (indentationVel + 1 / (1.5 * self.dissipation))) + 
               0.5 + eps2) * (0.5 * np.tanh(bd * indentation) + 0.5 + eps2) *
              fHd)
        force = fn * self.normal
        # Friction force.
        aux = ((vtangent[0, 0]) ** 2 + (vtangent[0, 1]) ** 2 + 
               (vtangent[0, 2]) ** 2 + eps)
        vslip = aux ** (0.5)
        vrel = vslip / self.transitionVelocity
        ffriction = fn * (np.fmin(vrel, 1) * (self.dynamicFriction + 
                          2 * (self.staticFriction - self.dynamicFriction) / 
                          (1 + vrel * vrel)) + self.viscousFriction * vslip)
        # Contact force.
        contactForce = force + ffriction * (-vtangent) / vslip
        
        return contactForce
