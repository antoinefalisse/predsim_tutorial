'''
    This script contains helper functions used in this project. Not all
    functions are still in use, but they might be in the future.
'''

# %% Import packages.
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import casadi as ca
import matplotlib.pyplot as plt  

# %% Storage file to numpy array.
# Found here: https://github.com/chrisdembia/perimysium/
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data
    
# %% Storage file to dataframe.
def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% Extract IK results from storage file.
def getIK(storage_file, joints, degrees=False):
    # Extract data
    data = storage2numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, joint in enumerate(joints):  
        if ((joint == 'pelvis_tx') or (joint == 'pelvis_ty') or 
            (joint == 'pelvis_tz')):
            Qs.insert(count + 1, joint, data[joint])         
        else:
            if degrees == True:
                Qs.insert(count + 1, joint, data[joint])                  
            else:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)              
            
    # Filter data    
    fs=1/np.mean(np.diff(Qs['time']))    
    fc = 6  # Cut-off frequency of the filter
    order = 4
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(b, a, Qs.loc[:, Qs.columns != 'time'], axis=0, 
                             padtype='odd', padlen=3*(max(len(b),len(a))-1))    
    output = pd.DataFrame(data=output, columns=joints)
    QsFilt = pd.concat([pd.DataFrame(data=data['time'], columns=['time']), 
                        output], axis=1)    
    
    return Qs, QsFilt

# %% Extract activations from storage file.
def getActivations(storage_file, muscles):
    # Extract data
    data = storage2numpy(storage_file)
    activations = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, muscle in enumerate(muscles):  
            activations.insert(count + 1, muscle, data[muscle + "activation"])              
                
    return activations

# %% Extract ground reaction forces from storage file.
def getGRF(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    GRFs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        GRFs.insert(count + 1, header, data[header])    
    
    return GRFs

# %% Extract ID results from storage file.
def getID(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        if ((header == 'pelvis_tx') or (header == 'pelvis_ty') or 
            (header == 'pelvis_tz')):
            out.insert(count + 1, header, data[header + '_force'])              
        else:
            out.insert(count + 1, header, data[header + '_moment'])    
    
    return out

# %% Extract from storage file.
def getFromStorage(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% Compute ground reaction moments (GRM).
def getGRM_wrt_groundOrigin(storage_file, fHeaders, pHeaders, mHeaders):
    # Extract data
    data = storage2numpy(storage_file)
    GRFs = pd.DataFrame()    
    for count, fheader in enumerate(fHeaders):
        GRFs.insert(count, fheader, data[fheader])  
    PoAs = pd.DataFrame()    
    for count, pheader in enumerate(pHeaders):
        PoAs.insert(count, pheader, data[pheader]) 
    GRMs = pd.DataFrame()    
    for count, mheader in enumerate(mHeaders):
        GRMs.insert(count, mheader, data[mheader])  
        
    # GRT_x = PoA_y*GRF_z - PoA_z*GRF_y
    # GRT_y = PoA_z*GRF_x - PoA_z*GRF_z + T_y
    # GRT_z = PoA_x*GRF_y - PoA_y*GRF_x
    GRM_wrt_groundOrigin = pd.DataFrame(data=data['time'], columns=['time'])    
    GRM_wrt_groundOrigin.insert(1, mHeaders[0], PoAs[pHeaders[1]] * GRFs[fHeaders[2]]  - PoAs[pHeaders[2]] * GRFs[fHeaders[1]])
    GRM_wrt_groundOrigin.insert(2, mHeaders[1], PoAs[pHeaders[2]] * GRFs[fHeaders[0]]  - PoAs[pHeaders[0]] * GRFs[fHeaders[2]] + GRMs[mHeaders[1]])
    GRM_wrt_groundOrigin.insert(3, mHeaders[2], PoAs[pHeaders[0]] * GRFs[fHeaders[1]]  - PoAs[pHeaders[1]] * GRFs[fHeaders[0]])        
    
    return GRM_wrt_groundOrigin

# %% Compite center of pressure (COP).
def getCOP(GRF, GRM):
    
    COP = np.zeros((3, GRF.shape[1]))
    torques = np.zeros((3, GRF.shape[1]))
    
    COP[0, :] = GRM[2, :] / GRF[1, :]    
    COP[2, :] = -GRM[0, :] / GRF[1, :]
    
    torques[1, :] = GRM[1, :] - COP[2, :]*GRF[0, :] + COP[0, :]*GRF[2, :]
    
    return COP, torques

# %% Get indices from list.
def getJointIndices(joints, selectedJoints):
    
    jointIndices = []
    for joint in selectedJoints:
        jointIndices.append(joints.index(joint))
            
    return jointIndices

# %% Get moment arm indices.
def getMomentArmIndices(rightMuscles, leftPolynomialJoints,
                        rightPolynomialJoints, polynomialData):
         
    momentArmIndices = {}
    for count, muscle in enumerate(rightMuscles):        
        spanning = polynomialData[muscle]['spanning']
        for i in range(len(spanning)):
            if (spanning[i] == 1):
                momentArmIndices.setdefault(
                        leftPolynomialJoints[i], []).append(count)
    for count, muscle in enumerate(rightMuscles):        
        spanning = polynomialData[muscle]['spanning']
        for i in range(len(spanning)):
            if (spanning[i] == 1):
                momentArmIndices.setdefault(
                        rightPolynomialJoints[i], []).append(
                                count + len(rightMuscles))                
        
    return momentArmIndices

# %% Solve OCP using bounds instead of constraints.
def solve_with_bounds(opti, tolerance):
    # Get guess
    guess = opti.debug.value(opti.x, opti.initial())
    # Sparsity pattern of the constraint Jacobian
    jac = ca.jacobian(opti.g, opti.x)
    sp = (ca.DM(jac.sparsity(), 1)).sparse()
    # Find constraints dependent on one variable
    is_single = np.sum(sp, axis=1)
    is_single_num = np.zeros(is_single.shape[0])
    for i in range(is_single.shape[0]):
        is_single_num[i] = np.equal(is_single[i, 0], 1)
    # Find constraints with linear dependencies or no dependencies
    is_nonlinear = ca.which_depends(opti.g, opti.x, 2, True)
    is_linear = [not i for i in is_nonlinear]
    is_linear_np = np.array(is_linear)
    is_linear_np_num = is_linear_np*1
    # Constraints dependent linearly on one variable should become bounds
    is_simple = is_single_num.astype(int) & is_linear_np_num
    idx_is_simple = [i for i, x in enumerate(is_simple) if x]
    ## Find corresponding variables
    col = np.nonzero(sp[idx_is_simple, :].T)[0]
    # Contraint values
    lbg = opti.lbg
    lbg = opti.value(lbg)
    ubg = opti.ubg
    ubg = opti.value(ubg)
    # Detect  f2(p)x+f1(p)==0
    # This is important if  you have scaled variables: x = 10*opti.variable()
    # with a constraint -10 < x < 10. Because in the reformulation we read out
    # the original variable and thus we need to scale the bounds appropriately.
    g = opti.g
    gf = ca.Function('gf', [opti.x, opti.p], [g[idx_is_simple, 0], 
                            ca.jtimes(g[idx_is_simple, 0], opti.x, 
                                      np.ones((opti.nx, 1)))])
    [f1, f2] = gf(0, opti.p)
    f1 = (ca.evalf(f1)).full() # maybe a problem here
    f2 = (ca.evalf(f2)).full()
    lb = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    ub = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    # Initialize bound vector
    lbx = -np.inf * np.ones((opti.nx))
    ubx = np.inf * np.ones((opti.nx))
    # Fill bound vector. For unbounded variables, we keep +/- inf.
    for i in range(col.shape[0]):
        lbx[col[i]] = np.maximum(lbx[col[i]], lb[i])
        ubx[col[i]] = np.minimum(ubx[col[i]], ub[i])      
    lbx[col] = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    ubx[col] = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    # Updated constraint value vector
    not_idx_is_simple = np.delete(range(0, is_simple.shape[0]), idx_is_simple)
    new_g = g[not_idx_is_simple, 0]
    # Updated bounds
    llb = lbg[not_idx_is_simple]
    uub = ubg[not_idx_is_simple]
    
    prob = {'x': opti.x, 'f': opti.f, 'g': new_g}
    s_opts = {}
    s_opts["expand"] = False
    s_opts["ipopt.hessian_approximation"] = "limited-memory"
    s_opts["ipopt.mu_strategy"] = "adaptive"
    s_opts["ipopt.max_iter"] = 10000
    s_opts["ipopt.tol"] = 10**(-tolerance)
    # s_opts["ipopt.print_frequency_iter"] = 100 
    solver = ca.nlpsol("solver", "ipopt", prob, s_opts)
    # Solve
    arg = {}
    arg["x0"] = guess
    # Bounds on x
    arg["lbx"] = lbx
    arg["ubx"] = ubx
    # Bounds on g
    arg["lbg"] = llb
    arg["ubg"] = uub    
    sol = solver(**arg) 
    # Extract and save results
    w_opt = sol['x'].full()
    stats = solver.stats()
    
    return w_opt, stats

# %% Solve OCP with constraints.
def solve_with_constraints(opti, tolerance):
    s_opts = {"hessian_approximation": "limited-memory",
              "mu_strategy": "adaptive",
              "max_iter": 2,
              "tol": 10**(-tolerance)}
    p_opts = {"expand":False}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()  
    
    return sol

# %% Write storage file from numpy array.    
def numpy_to_storage(labels, data, storage_file, datatype=None):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    # Old style
    if datatype is None:
        f = open(storage_file, 'w')
        f.write('name %s\n' %storage_file)
        f.write('datacolumns %d\n' %data.shape[1])
        f.write('datarows %d\n' %data.shape[0])
        f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
        f.write('endheader \n')
    # New style
    else:
        if datatype == 'IK':
            f.write('Coordinates\n')
        elif datatype == 'ID':
            f.write('Inverse Dynamics Generalized Forces\n')
        elif datatype == 'GRF':
            f.write('%s\n' %storage_file)
        elif datatype == 'muscle_forces':
            f.write('ModelForces\n')
        f.write('version=1\n')
        f.write('nRows=%d\n' %data.shape[0])
        f.write('nColumns=%d\n' %data.shape[1])    
        if datatype == 'IK':
            f.write('inDegrees=yes\n\n')
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n")
        elif datatype == 'ID':
            f.write('inDegrees=no\n')
        elif datatype == 'GRF':
            f.write('inDegrees=yes\n')
        elif datatype == 'muscle_forces':
            f.write('inDegrees=yes\n\n')
            f.write('This file contains the forces exerted on a model during a simulation.\n\n')
            f.write("A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n")
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write('Angles are in degrees.\n\n')
            
        f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close()
    
# %% Interpolate dataframe and return numpy array.    
def interpolateDataFrame2Numpy(dataFrame, tIn, tEnd, N):   
    
    tOut = np.linspace(tIn, tEnd, N)
    dataInterp = np.zeros([N, len(dataFrame.columns)])
    for i, col in enumerate(dataFrame.columns):
        set_interp = interp1d(dataFrame['time'], dataFrame[col])
        dataInterp[:,i] = set_interp(tOut)
        
    return dataInterp    

# %% Interpolate dataframe.
def interpolateDataFrame(dataFrame, tIn, tEnd, N):   
    
    tOut = np.linspace(tIn, tEnd, N)    
    dataInterp = pd.DataFrame() 
    for i, col in enumerate(dataFrame.columns):
        set_interp = interp1d(dataFrame['time'], dataFrame[col])        
        dataInterp.insert(i, col, set_interp(tOut))
        
    return dataInterp

# %% Scale dataframe.
def scaleDataFrame(dataFrame, scaling, headers):
    dataFrame_scaled = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_scaled.insert(count+1, header, 
                                dataFrame[header] / scaling.iloc[0][header])
        
    return dataFrame_scaled

# %% Unscale dataframe.
def unscaleDataFrame2(dataFrame, scaling, headers):
    dataFrame_scaled = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_scaled.insert(count+1, header, 
                                dataFrame[header] * scaling.iloc[0][header])
        
    return dataFrame_scaled

# %% Plot variables against their bounds.
def plotVSBounds(y,lb,ub,title=''):    
    ny = np.floor(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny+1), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.hlines(lb[i,0],x[0],x[-1],'r')
            ax.hlines(ub[i,0],x[0],x[-1],'b')
    plt.show()
         
# %% Plot variables against their bounds, which might be time-dependent.
def plotVSvaryingBounds(y,lb,ub,title=''):    
    ny = np.floor(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny+1), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.plot(x,lb[i,:],'r')
            ax.plot(x,ub[i,:],'b')
    plt.show()
            
# %% Plot paraeters.
def plotParametersVSBounds(y,lb,ub,title='',xticklabels=[]):    
    x = np.linspace(1,y.shape[0],y.shape[0])   
    plt.figure()
    ax = plt.gca()
    ax.scatter(x,lb,c='r',marker='_')
    ax.scatter(x,ub,c='b',marker='_')
    ax.scatter(x,y,c='k')
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels) 
    ax.set_title(title)
    
# %% Calculate number of subplots.
def nSubplots(N):
    
    ny_0 = (np.sqrt(N)) 
    ny = np.round(ny_0) 
    ny_a = int(ny)
    ny_b = int(ny)
    if (ny == ny_0) == False:
        if ny_a == 1:
            ny_b = N
        if ny < ny_0:
            ny_b = int(ny+1)
            
    return ny_a, ny_b

# %% Compute index initial contact from GRFs.
def getIdxIC_3D(GRF_opt, threshold):    
    idxIC = np.nan
    N = GRF_opt.shape[1]
    legIC = "undefined"    
    GRF_opt_rl = np.concatenate((GRF_opt[1,:], GRF_opt[4,:]))
    last_noContact = np.argwhere(GRF_opt_rl < threshold)[-1]
    if last_noContact == 2*N - 1:
        first_contact = np.argwhere(GRF_opt_rl > threshold)[0]
    else:
        first_contact = last_noContact + 1
    if first_contact >= N:
        idxIC = first_contact - N
        legIC = "left"
    else:
        idxIC = first_contact
        legIC = "right"
            
    return idxIC, legIC
      
# %% Compute RMSE.      
def getRMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# %% Compute RMSE normalized by signal range.
def getRMSENormMinMax(predictions, targets):    
    ROM = np.max(targets) - np.min(targets)    
    return (np.sqrt(((predictions - targets) ** 2).mean()))/ROM

# %% Compute RMSE normalized by standard deviation.
def getRMSENormStd(predictions, targets):    
    std = np.std(targets)
    return (np.sqrt(((predictions - targets) ** 2).mean()))/std

# %% Compute R2.
def getR2(predictions, targets):
    return (np.corrcoef(predictions, targets)[0,1])**2 

# %% Return some metrics.
def getMetrics(predictions, targets):
    r2 = np.zeros((predictions.shape[0]))
    rmse = np.zeros((predictions.shape[0]))
    rmseNormMinMax = np.zeros((predictions.shape[0]))
    rmseNormStd = np.zeros((predictions.shape[0]))
    for i in range(predictions.shape[0]):
        r2[i] = getR2(predictions[i,:], targets[i,:]) 
        rmse[i] = getRMSE(predictions[i,:],targets[i,:])  
        rmseNormMinMax[i] = getRMSENormMinMax(predictions[i,:],targets[i,:])   
        rmseNormStd[i] = getRMSENormStd(predictions[i,:],targets[i,:])        
    return r2, rmse, rmseNormMinMax, rmseNormStd

# %% Euler integration error.
def eulerIntegration(xk_0, xk_1, uk, delta_t):
    
    return (xk_1 - xk_0) - uk * delta_t

# %% Get initial contacts.
def getInitialContact(GRF_y, time, threshold):
    
    idxIC = np.argwhere(GRF_y >= threshold)[0]
    timeIC = time[idxIC]
    
    timeIC_round2 = np.round(timeIC, 2)
    idxIC_round2 = np.argwhere(time >= timeIC_round2)[0]
    
    return idxIC, timeIC, idxIC_round2, timeIC_round2    
