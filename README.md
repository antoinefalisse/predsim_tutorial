# 3D Predictive Simulations of Walking

This repository contains code and data to generate predictive simulations of human walking as described in [Falisse A, Afschrift M, De Groote F (2022) _Modeling toes contributes to realistic stance knee mechanics in three-dimensional predictive simulations of walking_. PLoS ONE 17(1): e0256311](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256311).

Here is an example of a predictive simulation of walking, based on a complex musculoskeletal model (31 degrees of freedom, 92 muscles, 6 compliant foot-ground contacts per foot), generated with our framework.

![Predictive simulation of human walking (doi:10.1098/rsif.2019.0402)](doc/images/PredictiveSimulation.gif)

The main script is `main.py` and the easiest is to start exploring the code from there. The code is for use on Windows, but please post an issue if you want support for other platforms as changes are minimal.

# Install requirements
- Open Anaconda prompt
- Create environment: `conda create -n 3dpredsim pip spyder`
- Activate environment: `activate 3dpredsim`
- Navigate to the folder where you want to download the code: eg. `cd Documents`
- Download code: `git clone https://github.com/antoinefalisse/predictsim_mtp.git`
- Navigate to the folder: `cd predictsim_tutorial`
- Install required packages: `python -m pip install -r requirements.txt`

# Overview of main files:
- `main.py`: basically everything from loading data, formulating the problem, solving it, and processing the results. Yes I know, it is not fantastic pratice to have everything in one place. Case 0 ([default](https://github.com/antoinefalisse/predsim_tutorial/blob/main/main.py#L51)) converges in 704 iterations on my windows machine. 
- `plotResults.py`: plots of simulation results against reference data (eg, joint angles and torques, ground reaction forces, and muscle activations).
- The other files are mainly containing classes and functions called in `main.py`.

# Visualize simulations in OpenSim
- Launch OpenSim
- Open model, eg `OpenSimModel/Hamner_modified/Model/Hamner_modified_scaled.osim.osim`
- Load motion, eg `Results/Case_0/motion.mot`
- Associate Motion Data, eg `Results/Case_0/GRF.mot`

# Citation
Please cite this paper:
  - Falisse A, Afschrift M, De Groote F (2022) _Modeling toes contributes to realistic stance knee mechanics in three-dimensional predictive simulations of walking_. PLoS ONE 17(1): e0256311. https://doi.org/10.1371/journal.pone.0256311
