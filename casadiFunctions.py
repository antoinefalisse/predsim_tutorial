'''
    This script contains several CasADi functions for use when setting up
    the optimal control problem.
'''

# %% Import packages.
import casadi as ca
import numpy as np

# %% CasADi function to approximate muscle-tendon lenghts, velocities,
# and moment arms based on joint positions and velocities.
def polynomialApproximation(musclesPolynomials, polynomialData, NPolynomial):    
    
    from polynomials import polynomials
    
    # Function variables.
    qin = ca.SX.sym('qin', 1, NPolynomial)
    qdotin  = ca.SX.sym('qdotin', 1, NPolynomial)
    lMT = ca.SX(len(musclesPolynomials), 1)
    vMT = ca.SX(len(musclesPolynomials), 1)
    dM = ca.SX(len(musclesPolynomials), NPolynomial)
    
    for count, musclePolynomials in enumerate(musclesPolynomials):
        
        coefficients = polynomialData[musclePolynomials]['coefficients']
        dimension = polynomialData[musclePolynomials]['dimension']
        order = polynomialData[musclePolynomials]['order']        
        spanning = polynomialData[musclePolynomials]['spanning']          
        
        polynomial = polynomials(coefficients, dimension, order)
        
        idxSpanning = [i for i, e in enumerate(spanning) if e == 1]        
        lMT[count] = polynomial.calcValue(qin[0, idxSpanning])
        
        dM[count, :] = 0
        vMT[count] = 0        
        for i in range(len(idxSpanning)):
            dM[count, idxSpanning[i]] = - polynomial.calcDerivative(
                    qin[0, idxSpanning], i)
            vMT[count] += (-dM[count, idxSpanning[i]] * 
               qdotin[0, idxSpanning[i]])
        
    f_polynomial = ca.Function('f_polynomial',[qin, qdotin],[lMT, vMT, dM])
    
    return f_polynomial
        
# %% CasADi function to derive the Hill equilibrium.
def hillEquilibrium(mtParameters, tendonCompliance, specificTension):
    
    from muscleModels import DeGrooteFregly2016MuscleModel
    
    NMuscles = mtParameters.shape[1]
    
    # Function variables.
    activation = ca.SX.sym('activation', NMuscles)
    mtLength = ca.SX.sym('mtLength', NMuscles)
    mtVelocity = ca.SX.sym('mtVelocity', NMuscles)
    normTendonForce = ca.SX.sym('normTendonForce', NMuscles)
    normTendonForceDT = ca.SX.sym('normTendonForceDT', NMuscles)
     
    hillEquilibrium = ca.SX(NMuscles, 1)
    tendonForce = ca.SX(NMuscles, 1)
    activeFiberForce = ca.SX(NMuscles, 1)
    normActiveFiberLengthForce = ca.SX(NMuscles, 1)
    passiveFiberForce = ca.SX(NMuscles, 1)
    normFiberLength = ca.SX(NMuscles, 1)
    fiberVelocity = ca.SX(NMuscles, 1)    
    
    for m in range(NMuscles):    
        muscle = DeGrooteFregly2016MuscleModel(
            mtParameters[:, m], activation[m], mtLength[m], mtVelocity[m], 
            normTendonForce[m], normTendonForceDT[m], tendonCompliance[:, m],
            specificTension[:, m])
        
        hillEquilibrium[m] = muscle.deriveHillEquilibrium()
        tendonForce[m] = muscle.getTendonForce()
        activeFiberForce[m] = muscle.getActiveFiberForce()[0]
        passiveFiberForce[m] = muscle.getPassiveFiberForce()[0]
        normActiveFiberLengthForce[m] = muscle.getActiveFiberLengthForce()
        normFiberLength[m] = muscle.getFiberLength()[1]
        fiberVelocity[m] = muscle.getFiberVelocity()[0]
        
    f_hillEquilibrium = ca.Function('f_hillEquilibrium',
                                    [activation, mtLength, mtVelocity, 
                                     normTendonForce, normTendonForceDT], 
                                     [hillEquilibrium, tendonForce,
                                      activeFiberForce, passiveFiberForce,
                                      normActiveFiberLengthForce,
                                      normFiberLength, fiberVelocity]) 
    
    return f_hillEquilibrium

# %% CasADi function to explicitly describe the dynamic equations governing 
# the arm movements.
def armActivationDynamics(NArmJoints):
    
    t = 0.035 # time constant       
    
    # Function variables.
    eArm = ca.SX.sym('eArm',NArmJoints)
    aArm = ca.SX.sym('aArm',NArmJoints)
    
    aArmDt = (eArm - aArm) / t
    
    f_armActivationDynamics = ca.Function('f_armActivationDynamics',
                                          [eArm, aArm], [aArmDt])
    
    return f_armActivationDynamics  

# %% CasADi function to compute the metabolic cost of transport based on 
# Bhargava et al. (2004).
def metabolicsBhargava(slowTwitchRatio, maximalIsometricForce,
                       muscleMass, smoothingConstant,
                       use_fiber_length_dep_curve=False,
                       use_force_dependent_shortening_prop_constant=True,
                       include_negative_mechanical_work=False):
    
    NMuscles = maximalIsometricForce.shape[0]
    
    # Function variables.
    excitation = ca.SX.sym('excitation', NMuscles)
    activation = ca.SX.sym('activation', NMuscles)
    normFiberLength = ca.SX.sym('normFiberLength', NMuscles)
    fiberVelocity = ca.SX.sym('fiberVelocity', NMuscles)
    activeFiberForce = ca.SX.sym('activeFiberForce', NMuscles)
    passiveFiberForce = ca.SX.sym('passiveFiberForce', NMuscles)
    normActiveFiberLengthForce = (
            ca.SX.sym('normActiveFiberLengthForce', NMuscles))
    
    activationHeatRate = ca.SX(NMuscles, 1)
    maintenanceHeatRate = ca.SX(NMuscles, 1)
    shorteningHeatRate = ca.SX(NMuscles, 1)
    mechanicalWork = ca.SX(NMuscles, 1)
    totalHeatRate = ca.SX(NMuscles, 1) 
    metabolicEnergyRate = ca.SX(NMuscles, 1) 
    slowTwitchExcitation = ca.SX(NMuscles, 1) 
    fastTwitchExcitation = ca.SX(NMuscles, 1) 
    
    from metabolicEnergyModels import Bhargava2004SmoothedMuscleMetabolics
    
    for m in range(NMuscles):   
        metabolics = (Bhargava2004SmoothedMuscleMetabolics(
            excitation[m], activation[m], 
                                         normFiberLength[m],
                                         fiberVelocity[m],
                                         activeFiberForce[m], 
                                         passiveFiberForce[m],
                                         normActiveFiberLengthForce[m],
                                         slowTwitchRatio[m], 
                                         maximalIsometricForce[m],
                                         muscleMass[m], smoothingConstant))
        
        slowTwitchExcitation[m] = metabolics.getTwitchExcitation()[0] 
        fastTwitchExcitation[m] = metabolics.getTwitchExcitation()[1] 
        activationHeatRate[m] = metabolics.getActivationHeatRate()        
        maintenanceHeatRate[m] = metabolics.getMaintenanceHeatRate(
                use_fiber_length_dep_curve)        
        shorteningHeatRate[m] = metabolics.getShorteningHeatRate(
                use_force_dependent_shortening_prop_constant)        
        mechanicalWork[m] = metabolics.getMechanicalWork(
                include_negative_mechanical_work)        
        totalHeatRate[m] = metabolics.getTotalHeatRate()
        metabolicEnergyRate[m] = metabolics.getMetabolicEnergyRate()
    
    f_metabolicsBhargava = ca.Function('metabolicsBhargava',
                                    [excitation, activation, normFiberLength, 
                                     fiberVelocity, activeFiberForce, 
                                     passiveFiberForce, 
                                     normActiveFiberLengthForce], 
                                     [activationHeatRate, maintenanceHeatRate,
                                      shorteningHeatRate, mechanicalWork, 
                                      totalHeatRate, metabolicEnergyRate])
    
    return f_metabolicsBhargava

# %% CasADi function to compute passive (limit) joint torques.
def getLimitTorques(k, theta, d):
    
    # Function variables.
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = (k[0] * np.exp(k[1] * (Q - theta[1])) + k[2] * 
                           np.exp(k[3] * (Q - theta[0])) - d * Qdot)
    
    f_passiveJointTorque = ca.Function('f_passiveJointTorque', [Q, Qdot], 
                                       [passiveJointTorque])
    
    return f_passiveJointTorque

# %% CasADi function to compute linear passive joint torques.
def getLinearPassiveTorques(k, d):
    
    # Function variables.
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = -k * Q - d * Qdot
    f_passiveMtpTorque = ca.Function('f_passiveMtpTorque', [Q, Qdot], 
                                     [passiveJointTorque])
    
    return f_passiveMtpTorque    

# %% CasADi function to compute normalized sum of elements to given power.
def normSumPow(N, exp):
    
    # Function variables.
    x = ca.SX.sym('x', N,  1)
      
    nsp = ca.sum1(x**exp)       
    nsp = nsp / N
    
    f_normSumPow = ca.Function('f_normSumPow', [x], [nsp])
    
    return f_normSumPow

# %% CasADi function to compute difference in torques.
def diffTorques():
    
    # Function variables.
    jointTorque = ca.SX.sym('x', 1) 
    muscleTorque = ca.SX.sym('x', 1) 
    passiveTorque = ca.SX.sym('x', 1)
    
    diffTorque = jointTorque - (muscleTorque + passiveTorque)
    
    f_diffTorques = ca.Function(
            'f_diffTorques', [jointTorque, muscleTorque, passiveTorque], 
            [diffTorque])
        
    return f_diffTorques

# %% CasADi function to compute foot-ground contact forces.
# Note: this function is unused for the predictive simulations, but could be
# useful in other studies.
def smoothSphereHalfSpaceForce(dissipation, transitionVelocity,
                               staticFriction, dynamicFriction, 
                               viscousFriction, normal):
    
    from contactModels import smoothSphereHalfSpaceForce_ca
    
    # Function variables.
    stiffness = ca.SX.sym('stiffness', 1) 
    radius = ca.SX.sym('radius', 1)     
    locSphere_inB = ca.SX.sym('locSphere_inB', 3) 
    posB_inG = ca.SX.sym('posB_inG', 3) 
    lVelB_inG = ca.SX.sym('lVelB_inG', 3) 
    aVelB_inG = ca.SX.sym('aVelB_inG', 3) 
    RBG_inG = ca.SX.sym('RBG_inG', 3, 3) 
    TBG_inG = ca.SX.sym('TBG_inG', 3) 
    
    contactElement = smoothSphereHalfSpaceForce_ca(
        stiffness, radius, dissipation, transitionVelocity, staticFriction,
        dynamicFriction, viscousFriction, normal)
    
    contactForce = contactElement.getContactForce(locSphere_inB, posB_inG,
                                                  lVelB_inG, aVelB_inG,
                                                  RBG_inG, TBG_inG)
    
    f_smoothSphereHalfSpaceForce = ca.Function(
            'f_smoothSphereHalfSpaceForce',[stiffness, radius, locSphere_inB,
                                            posB_inG, lVelB_inG, aVelB_inG,
                                            RBG_inG, TBG_inG], [contactForce])
    
    return f_smoothSphereHalfSpaceForce 
