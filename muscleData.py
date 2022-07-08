'''
    This script contains muscle-specific functions.
'''

# %% Import packages.
import os
import numpy as np

# %% Import muscle-tendon parameters.
# We save the muscle-tendon parameters associated with the model the first time
# we 'use' the model such that we do not need OpenSim later on. If no
# muscle-tendon parameters exist, then we extract them from the model using
# OpenSim's Python API. See here how to setup your environment to use the
# Python API: https://simtk-confluence.stanford.edu/display/OpenSim/Scripting+in+Python.
def getMTParameters(pathModel, muscles, loadMTParameters, modelName,
                    pathMTParameters=0):
    
    if loadMTParameters:        
        mtParameters = np.load(os.path.join(
            pathMTParameters, 'mtParameters_{}.npy'.format(modelName)), 
            allow_pickle=True)        
    else:
        import opensim
        model = opensim.Model(pathModel)
        mtParameters = np.zeros([5,len(muscles)])
        model_muscles = model.getMuscles()
        for i in range(len(muscles)):
           muscle = model_muscles.get(muscles[i])
           mtParameters[0,i] = muscle.getMaxIsometricForce()
           mtParameters[1,i] = muscle.getOptimalFiberLength()
           mtParameters[2,i] = muscle.getTendonSlackLength()
           mtParameters[3,i] = muscle.getPennationAngleAtOptimalFiberLength()
           mtParameters[4,i] = (muscle.getMaxContractionVelocity() * 
                                muscle.getOptimalFiberLength())
        if pathMTParameters != 0:
           np.save(
               os.path.join(pathMTParameters,
                            'mtParameters_{}.npy'.format(modelName)),
               mtParameters)
       
    return mtParameters

# %% Extract muscle-tendon lenghts and moment arms.
# We extract data from varying limb postures, such as to later fit polynomials
# to approximate muscle tendon lenghts and moment arms.
def get_mtu_length_and_moment_arm(pathModel, data, coordinates_table):
    import opensim
    
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathModel)  
    state = model.initSystem()    
    # muscles as ordered in model
    muscles = []
    forceSet = model.getForceSet()
    for i in range(forceSet.getSize()):        
        c_force_elt = forceSet.get(i)  
        if c_force_elt.getConcreteClassName() == "Thelen2003Muscle":
            muscles.append(c_force_elt.getName())
    lumbarMuscles = ['ercspn', 'intobl', 'extobl']
    
    # coordinates as ordered in model
    coordinateSet = model.getCoordinateSet()
    coordinates = [coordinateSet.get(i).getName() 
                   for i in range(coordinateSet.getSize())]
    lumbarCoordinates = ['lumbar_extension', 'lumbar_bending', 
                         'lumbar_rotation']
    coordinates_table_short = [
        label.split('/')[-2] for label in coordinates_table] # w/o /jointset/..

    lMT = np.zeros((data.shape[0], len(muscles)))
    dM =  np.zeros((data.shape[0], len(muscles), len(coordinates_table_short)))
    for i in range(data.shape[0]):
        for coordinate in coordinates_table:
            value_q = data[i, coordinates_table.index(coordinate)] * np.pi/180
            model.setStateVariableValue(state, coordinate, value_q)
        state = model.updWorkingState()
        model.realizePosition(state)        
        for m in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(m)  
            if c_force_elt.getConcreteClassName() == "Thelen2003Muscle":
                muscleName = c_force_elt.getName()
                cObj = opensim.Thelen2003Muscle.safeDownCast(c_force_elt)            
                lMT[i,m] = cObj.getLength(state)            
                for c, coord in enumerate(coordinates_table_short):
                    # We do not want to compute moment arms that are not
                    # relevant, eg for a muscle of the left side wrt a
                    # coordinate of the right side, or for a leg muscle with
                    # respect to a lumbar coordinate.
                    if muscleName[-2:] == '_l' and coord[-2:] == '_r':
                        dM[i, m, c] = 0
                    elif muscleName[-2:] == '_r' and coord[-2:] == '_l':
                        dM[i, m, c] = 0
                    elif (muscleName[:-2] in lumbarMuscles and 
                          not coord in lumbarCoordinates):
                        dM[i, m, c] = 0
                    elif (not muscleName[:-2] in lumbarMuscles and 
                          coord in lumbarCoordinates):
                        dM[i, m, c] = 0
                    else:
                        coordinate = coordinateSet.get(
                            coordinates.index(coord))
                        dM[i, m, c] = cObj.computeMomentArm(state, coordinate)
                        
    return [lMT, dM]

# %% Import data from polynomial approximations.
# We fit the polynomial coefficients if no polynomial data exist yet, and we
# save them such that we do not need to do the fitting again.
def getPolynomialData(loadPolynomialData, pathModelFolder, modelName,
                      pathMotionFile4Polynomials='', joints=[],
                      muscles=[], threshold=0.002, nThreads=None,
                      overwritedata4PolynomialFitting=False):
    
    if loadPolynomialData:
        polynomialData = np.load(
            os.path.join(
                pathModelFolder, 'polynomialData_{}.npy'.format(modelName)), 
            allow_pickle=True)         
    else:        
        path_data4PolynomialFitting = os.path.join(
            pathModelFolder, 'data4PolynomialFitting_{}.npy'.format(modelName))
        # Generate polynomial data
        if (not os.path.exists(path_data4PolynomialFitting) or 
            overwritedata4PolynomialFitting):
            import opensim
            from joblib import Parallel, delayed
            import multiprocessing
            # Get training data from motion file
            table = opensim.TimeSeriesTable(pathMotionFile4Polynomials)
            coordinates_table = list(table.getColumnLabels()) # w/ jointset/...
            data = table.getMatrix().to_numpy() # data in degrees w/o time        
            pathModel = os.path.join(pathModelFolder, modelName + '.osim')
            # Set number of threads
            if nThreads == None:
                nThreads = multiprocessing.cpu_count()-2 # default
            if nThreads < 1:
                nThreads = 1
            elif nThreads > multiprocessing.cpu_count():
                nThreads = multiprocessing.cpu_count()                
            # Generate muscle tendon lengths and moment arms (in parallel)
            slice_size = int(np.floor(data.shape[0]/nThreads))
            rest = data.shape[0] % nThreads 
            outputs = Parallel(n_jobs=nThreads)(
                delayed(get_mtu_length_and_moment_arm)(
                    pathModel, data[i*slice_size:(i+1)*slice_size,:], 
                    coordinates_table) for i in range(nThreads))
            if rest != 0:
                output_last = get_mtu_length_and_moment_arm(
                    pathModel, data[-rest:,:], coordinates_table)                
            # Gather data
            lMT = np.zeros((data.shape[0], outputs[0][1].shape[1]))
            dM =  np.zeros((data.shape[0], outputs[0][1].shape[1], 
                            outputs[0][1].shape[2]))
            for i in range(len(outputs)):
                lMT[i*slice_size:(i+1)*slice_size, :] = outputs[i][0]
                dM[i*slice_size:(i+1)*slice_size, :, :] = outputs[i][1]
            if rest != 0:
                lMT[-rest:, :] = output_last[0]
                dM[-rest:, :, :] = output_last[1]
            # Put data in dict
            # muscles as ordered in model
            opensim.Logger.setLevelString('error')
            model = opensim.Model(pathModel)  
            allMuscles = []
            forceSet = model.getForceSet()
            for i in range(forceSet.getSize()):        
                c_force_elt = forceSet.get(i)  
                if c_force_elt.getConcreteClassName() == "Thelen2003Muscle":
                    allMuscles.append(c_force_elt.getName())    
            data4PolynomialFitting = {}
            data4PolynomialFitting['mtu_lengths'] = lMT
            data4PolynomialFitting['mtu_moment_arms'] = dM
            data4PolynomialFitting['muscle_names'] = allMuscles
            data4PolynomialFitting['coordinate_names'] = [
                label.split('/')[-2] for label in coordinates_table]
            data4PolynomialFitting['coordinate_values'] = data
            # Save data
            np.save(path_data4PolynomialFitting, data4PolynomialFitting)
        else:
            data4PolynomialFitting = np.load(path_data4PolynomialFitting, 
                                             allow_pickle=True).item()
        # Fit polynomial coefficients        
        from polynomials import getPolynomialCoefficients
        polynomialData = getPolynomialCoefficients(
            data4PolynomialFitting, joints, muscles, threshold=threshold)
        if pathModelFolder != 0:
            np.save(os.path.join(pathModelFolder, 
                                 'polynomialData_{}.npy'.format(modelName)),
                    polynomialData)
           
    return polynomialData

# %% Tendon stiffness
# Default value is 35.
def tendonStiffness(NSideMuscles):
    tendonStiffness = np.full((1, NSideMuscles), 35)
    
    return tendonStiffness

# Tendon shift to ensure that the tendon force, when the normalized tendon
# lenght is 1, is the same for all tendon stiffnesses.
def tendonShift(NSideMuscles):
    tendonShift = np.full((1, NSideMuscles), 0)
    
    return tendonShift 

# %% Specific tensions from https://simtk.org/projects/idealassist_run
# Associated publication: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0163417
def specificTension(muscles):    
    
    sigma = {'glut_med1_r' : 0.74455,
             'glut_med2_r': 0.75395, 
             'glut_med3_r': 0.75057, 
             'glut_min1_r': 0.75, 
             'glut_min2_r': 0.75, 
             'glut_min3_r': 0.75116, 
             'semimem_r': 0.62524, 
             'semiten_r': 0.62121, 
             'bifemlh_r': 0.62222,
             'bifemsh_r': 1.00500, 
             'sar_r': 0.74286,
             'add_long_r': 0.74643, 
             'add_brev_r': 0.75263,
             'add_mag1_r': 0.55217,
             'add_mag2_r': 0.55323, 
             'add_mag3_r': 0.54831, 
             'tfl_r': 0.75161,
             'pect_r': 0.76000, 
             'grac_r': 0.73636, 
             'glut_max1_r': 0.75395, 
             'glut_max2_r': 0.74455, 
             'glut_max3_r': 0.74595, 
             'iliacus_r': 1.2477,
             'psoas_r': 1.5041,
             'quad_fem_r': 0.74706, 
             'gem_r': 0.74545, 
             'peri_r': 0.75254, 
             'rect_fem_r': 0.74936, 
             'vas_med_r': 0.49961, 
             'vas_int_r': 0.55263, 
             'vas_lat_r': 0.50027,
             'med_gas_r': 0.69865, 
             'lat_gas_r': 0.69694, 
             'soleus_r': 0.62703,
             'tib_post_r': 0.62520, 
             'flex_dig_r': 0.5, 
             'flex_hal_r': 0.50313,
             'tib_ant_r': 0.75417, 
             'per_brev_r': 0.62143,
             'per_long_r': 0.62450, 
             'per_tert_r': 1.0,
             'ext_dig_r': 0.75294,
             'ext_hal_r': 0.73636, 
             'ercspn_r': 0.25, 
             'intobl_r': 0.25, 
             'extobl_r': 0.25}
    
    specificTension = np.empty((1, len(muscles)))    
    for count, muscle in enumerate(muscles):
        specificTension[0, count] = sigma[muscle]
    
    return specificTension

# %% Slow twitch ratios from https://simtk.org/projects/idealassist_run
# Associated publication: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0163417
def slowTwitchRatio(muscles):    
    
    sigma = {'glut_med1_r' : 0.55,
             'glut_med2_r': 0.55, 
             'glut_med3_r': 0.55, 
             'glut_min1_r': 0.55, 
             'glut_min2_r': 0.55, 
             'glut_min3_r': 0.55, 
             'semimem_r': 0.4925, 
             'semiten_r': 0.425, 
             'bifemlh_r': 0.5425,
             'bifemsh_r': 0.529, 
             'sar_r': 0.50,
             'add_long_r': 0.50, 
             'add_brev_r': 0.50,
             'add_mag1_r': 0.552,
             'add_mag2_r': 0.552, 
             'add_mag3_r': 0.552, 
             'tfl_r': 0.50,
             'pect_r': 0.50, 
             'grac_r': 0.50, 
             'glut_max1_r': 0.55, 
             'glut_max2_r': 0.55, 
             'glut_max3_r': 0.55, 
             'iliacus_r': 0.50,
             'psoas_r': 0.50,
             'quad_fem_r': 0.50, 
             'gem_r': 0.50, 
             'peri_r': 0.50, 
             'rect_fem_r': 0.3865, 
             'vas_med_r': 0.503, 
             'vas_int_r': 0.543, 
             'vas_lat_r': 0.455,
             'med_gas_r': 0.566, 
             'lat_gas_r': 0.507, 
             'soleus_r': 0.803,
             'tib_post_r': 0.60, 
             'flex_dig_r': 0.60, 
             'flex_hal_r': 0.60,
             'tib_ant_r': 0.70, 
             'per_brev_r': 0.60,
             'per_long_r': 0.60, 
             'per_tert_r': 0.75,
             'ext_dig_r': 0.75,
             'ext_hal_r': 0.75, 
             'ercspn_r': 0.60,
             'intobl_r': 0.56, 
             'extobl_r': 0.58}
    
    slowTwitchRatio = np.empty((1, len(muscles)))    
    for count, muscle in enumerate(muscles):
        slowTwitchRatio[0, count] = sigma[muscle]
    
    return slowTwitchRatio

# %% Joint passive / limit torques.
# Data from https://www.tandfonline.com/doi/abs/10.1080/10255849908907988
def passiveTorqueData(joint):    
    
    kAll = {'hip_flexion_r' : [-2.44, 5.05, 1.51, -21.88],
            'hip_adduction_r': [-0.03, 14.94, 0.03, -14.94], 
            'hip_rotation_r': [-0.03, 14.94, 0.03, -14.94],
            'knee_angle_r': [-6.09, 33.94, 11.03, -11.33],
            'ankle_angle_r': [-2.03, 38.11, 0.18, -12.12],
            'subtalar_angle_r': [-60.21, 16.32, 60.21, -16.32],
            'mtp_angle_r': [-0.9, 14.87, 0.18, -70.08],
            'hip_flexion_l' : [-2.44, 5.05, 1.51, -21.88],
            'hip_adduction_l': [-0.03, 14.94, 0.03, -14.94], 
            'hip_rotation_l': [-0.03, 14.94, 0.03, -14.94],
            'knee_angle_l': [-6.09, 33.94, 11.03, -11.33],
            'ankle_angle_l': [-2.03, 38.11, 0.18, -12.12],
            'subtalar_angle_l': [-60.21, 16.32, 60.21, -16.32],
            'mtp_angle_l': [-0.9, 14.87, 0.18, -70.08],
            'lumbar_extension': [-0.35, 30.72, 0.25, -20.36],
            'lumbar_bending': [-0.25, 20.36, 0.25, -20.36],
            'lumbar_rotation': [-0.25, 20.36, 0.25, -20.36]}
    
    thetaAll = {'hip_flexion_r' : [-0.6981, 1.81],
                'hip_adduction_r': [-0.5, 0.5], 
                'hip_rotation_r': [-0.92, 0.92],
                'knee_angle_r': [-2.4, 0.13],
                'ankle_angle_r': [-0.74, 0.52],
                'subtalar_angle_r': [-0.65, 0.65],
                'mtp_angle_r': [0, 1.134464013796314],
                'hip_flexion_l' : [-0.6981, 1.81],
                'hip_adduction_l': [-0.5, 0.5], 
                'hip_rotation_l': [-0.92, 0.92],
                'knee_angle_l': [-2.4, 0.13],
                'ankle_angle_l': [-0.74, 0.52],
                'subtalar_angle_l': [-0.65, 0.65],
                'mtp_angle_l': [0, 1.134464013796314],
                'lumbar_extension': [-0.5235987755982988, 0.17],
                'lumbar_bending': [-0.3490658503988659, 0.3490658503988659],
                'lumbar_rotation': [-0.3490658503988659, 0.3490658503988659]}
    
    k = kAll[joint] 
    theta = thetaAll[joint]
    
    return k, theta
    