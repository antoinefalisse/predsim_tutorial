# Tutorial: 3D Predictive Simulations of Walking

This tutorial aims to guide users through the steps required to generate 3D predictive simulations of walking while leveraging [OpenSimAD](https://github.com/antoinefalisse/opensimAD). OpenSimAD is a custom version of OpenSim that supports algorithmic differentiation.

The generated predictive simulations should look like this:
<p align="center">
  <img src="doc/images/PredictiveSimulation.gif">
</p>

# Setup conda environment
1. Install [Anaconda](https://www.anaconda.com/)
2. Open Anaconda prompt
3. Create environment (python 3.9 recommended): `conda create -n predsim_tutorial python=3.9`
4. Activate environment: `conda activate predsim_tutorial`
5. Install OpenSim: `conda install -c opensim-org opensim=4.4=py39np120`
	- Test that OpenSim was successfully installed:
		- Start python: `python`
		- Import OpenSim: `import opensim`
			- If you don't get any error message at this point, you should be good to go.
		- You can also double check which version you installed : `opensim.GetVersion()`
		- Exit python: `quit()`
	- Visit this [webpage](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Conda+Package) for more details about the OpenSim conda package.
6. (Optional): Install an IDE such as Spyder: `conda install spyder`
7. Clone the repository to your machine: 
	- Navigate to the directory where you want to download the code: eg. `cd ./Documents`
	- Clone the repository: `git clone https://github.com/antoinefalisse/predsim_tutorial.git`
	- Navigate to the directory: `cd predsim_tutorial`
8. Install required packages: `python -m pip install -r requirements.txt`

# Tutorial
## Part 1: Generate external function
(You can skip to part 2 if you are only interested in the simulation part, the outputs from part 1 are already available).

To leverage the benefits of algorithmic differentiation, we use [CasADi external functions](https://web.casadi.org/docs/#casadi-s-external-function). In our case, the external functions typically take as inputs the multi-body model states (joint positions and speeds) and controls (joint accelerations) and return the joint torques after solving inverse dynamics. To generate the external function, we will need our OpenSim musculoskeletal model and we will use [OpenSimAD](https://github.com/antoinefalisse/opensimAD). In this tutorial, the OpenSim model we will use is a scaled version of the Hamner model, you can find it here: /Documents/predsim_tutorial/OpenSimModel/Hamner_modified/Model/Hamner_modified_scaled.osim. Let's first generate the external function corresponding to this model.
1. Install [OpenSimAD](https://github.com/antoinefalisse/opensimAD) following the instructions in the [README](https://github.com/antoinefalisse/opensimAD#install-requirements). We will assume you downloaded the repository under /Documents/opensimAD.
2. Copy the scaled musculoskeletal model (Hamner_modified_scaled.osim) in /Documents/opensimAD/examples (fyi there should already be two example models in that folder).
3. In /Documents/opensimAD/main.py, adjust [`modelName`](https://github.com/antoinefalisse/opensimAD/blob/main/main.py#L46) to the name of your scaled model: `modelName = 'Hamner_modified_scaled'`.
4. Run `main.py` (make sure you activated the opensim-ad conda environment). You should see some new files in /Documents/opensimAD/examples. Among them, the following three files: `Hamner_modified_scaled.cpp`, `Hamner_modified_scaled.npy`, and `Hamner_modified_scaled.dll` (Windows) or `Hamner_modified_scaled.so` (Linux) or `Hamner_modified_scaled.dylib` (macOS).
5. Create a folder ExternalFunction in /Documents/predsim_tutorial/OpenSimModel/Hamner_modified/Model/ and copy the three files mentioned above in this new folder (yes they are already there in case you wanted to skip part 1). It should look like:
	- /Documents/predsim_tutorial/OpenSimModel/Hamner_modified/Model/ExternalFunction/Hamner_modified_scaled.cpp
	- /Documents/predsim_tutorial/OpenSimModel/Hamner_modified/Model/ExternalFunction/Hamner_modified_scaled.npy
	- /Documents/predsim_tutorial/OpenSimModel/Hamner_modified/Model/ExternalFunction/Hamner_modified_scaled.dll (or .so or .dylib)

## Part 2: Generate simulations
In this part, we will generate the predictive simulations. We use direct collocation methods for formulating the problem (you can find more details about these methods in the publications below). We simulate for half a gait cycle, and impose left-right symmetricity such that we can reconstruct a full gait cycle post-optimization. Let's first generate the simulations with the model for which we generated the external function in part 1 (if you skipped part 1, no worries the required files were pre-generated).
1. **Set settings.** In `settings.py`, we will set some settings for the simulations (this is already done). We will run the case '0' to start with (key '0' in the settings dictionnary). We want to use the model 'Hamner_modified' (which is the one for which we generated the external function in part 1). Other settings are the target speed, let's set it to 1.33m/s, and the number of mesh intervals, let's use 25 mesh intervals. FYI since we use a third-order collocation scheme, the dynamic equations are enforced at three collocation points within each interval. With 25 mesh intervals, this means we have 75 collocation points. Assuming half a gait cycle is about 0.55s, we therefore enforce the dynamic constraints about every 7ms.
2. **Run simulation.** In `main.py`, let's now run the case '0'. You can select which case to run by adjusting the [list cases](https://github.com/antoinefalisse/predsim_tutorial/blob/tutorial/main.py#L38). By default, it will run the case '0'.
	- Run `main.py` either in the terminal (`python main.py` or in your favorite IDE).
	- The first time you run `main.py` with a new model, the script will approximate polynomial expressions to estimate muscle-tendon lenghts, velocities, and moment arms from joint coordinate values and speeds. It will also save the model body mass and the muscle-tendon parameters as .npy files (see the few files that appeared in the Model folder). After the polynomial fitting, you should see the Ipopt outputs on your console. That means the optimization problem is 'being solved'. If it succesfully converges, you should see "EXIT: Optimal Solution Found" after a little while.
3. **Visualize the results.** In the results folder, you should now see a new folder named case_0. This folder contains the results. The motion.mot file contains the optimal joint coordinate values and muscle activations, the GRF.mot file contains the resultant ground reaction forces, the stats.npy file contains some CasADi statistics about the solved optimization problem, and the w_opt.npy file contains the optimized design variables. Under /Documents/predsim_tutorial/Results/, you should also see a file named optimaltrajectories.npy, this file contains more results from the optimization (eg, metabolic cost of transport).
	- To visualize the optimal motion, you can use the OpenSim GUI. Open the model (Hamner_modified_scaled.osim), load the motion file (motion.mot), and associate the GRF data (GRF.mot). You should see something similar to the gif in this README.
	- To visualize the results, you can use the `plotResults.py` script. Set the case you want to visualize in the list cases at the top: `cases = ['0']` and run the script. You should see some more plots about joint coordinate values, muscle activations, ground reaction forces, etc.
	
You have generated a baseline simulation, congrats! Now let's make a few changes to the model and compare the results.
1. **Change the target speed.** In `settings.py`, create a new case '1' and change the target speed (this is already done). Now run `main.py` after making sure you will be running case '1'. Once the problem has converged, compare the results using the `plotResults.py` script. Set the list cases at the top to `cases = ['0', '1']`. What is the influence of walking at a slower speed?
2. **Change the strength of the glutei.** Create a new folder under Documents/predsim_tutorial/OpenSimModel and name it Hamner_modified_weakerGluts. Create a sub-folder Model. Copy the baseline model (Hamner_modified_scaled.osim) in the Model folder, rename it Hamner_modified_weakerGluts.osim and adjust the max_isometric_force of the glutei. In what we prepared for you, we halved the max forces of all of them (9 muscles per leg). Now follow the steps from Part 1 (technicaly this is not needed because you are not changing anything to the multi-body model, only to the muscles which are hard coded in this repository, but this makes things simpler in terms of paths etc). You should now have the three files from part 1 copied in Documents/predsim_tutorial/OpenSimModel/Hamner_modified_weakerGluts/Model/ExternalFunction. In `settings.py`, create a new case '2'. Adjust the model name to Hamner_modified_weakerGluts. Now run `main.py` after making sure you will be running case '2'. Once the problem has converged, compare the results using the `plotResults.py` script. Set the list cases at the top to `cases = ['0', '2']`. What is the influence of weakening the glutei?
3. **Change the stiffness of the contact spheres.** Create a new folder under Documents/predsim_tutorial/OpenSimModel and name it Hamner_modified_stifferContacts. Create a sub-folder Model. Copy the baseline model (Hamner_modified_scaled.osim) in the Model folder, rename it Hamner_modified_stifferContacts.osim and adjust the stiffness of the SmoothSphereHalfSpaceForces. In what we prepared for you, we set the stiffness to 10e6 instead of 1e5 for all 12 spheres (6 per foot). Now follow the steps from Part 1 (this is needed, the contact models are "part of the multi-body model"). You should now have the three files from part 1 copied in Documents/predsim_tutorial/OpenSimModel/Hamner_modified_stifferContacts/Model/ExternalFunction. In `settings.py`, create a new case '3'. Adjust the model name to Hamner_modified_stifferContacts. Now run `main.py` after making sure you will be running case '3'. Once the problem has converged, compare the results using the `plotResults.py` script. Set the list cases at the top to `cases = ['0', '3']`. What is the influence of using stiffer contact spheres?

# Visualize simulations in OpenSim
1. Launch OpenSim GUI
2. Open model, eg `OpenSimModel/Hamner_modified/Model/Hamner_modified_scaled.osim`
3. Load motion, eg `Results/Case_0_ref/motion.mot`
4. Associate Motion Data, eg `Results/Case_0_ref/GRF.mot`

# Limitations
We made some assumptions for the examples of this tutorial (eg, we assume the musculoskeletal model to be left-right symmetrical). Make sure you verify what you are doing if you end up using this code beyong the provided examples.

# Citation
This work is covered in three publications. Please consider citing these papers:
1. Study about supporting algorithmic differentation in OpenSim:
	- Falisse A, Serrancolí G, et al. (2019) Algorithmic differentiation improves the computational efficiency of OpenSim-based trajectory optimization of human movement. PLoS ONE 14(10): e0217730. https://doi.org/10.1371/journal.pone.0217730
2. Study about using predictive simulations to investigate different cost functions:
	- Falisse A, et al. (2019) Rapid predictive simulations with complex musculoskeletal models suggest that diverse healthy and pathological human gaits can emerge from similar control strategies. J. R. Soc. Interface.162019040220190402. http://doi.org/10.1098/rsif.2019.0402
3. Study about using predictive simulations to investigate the effect of mechanical assumptions:
	- Falisse A, Afschrift M, De Groote F (2022) _Modeling toes contributes to realistic stance knee mechanics in three-dimensional predictive simulations of walking. PLoS ONE 17(1): e0256311. https://doi.org/10.1371/journal.pone.0256311
