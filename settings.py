def getSettings():
        
    settings = {        
        # Baseline
        '0': {
            'model': 'Hamner_modified',
            'targetSpeed': 1.33,
            'N': 25},
        # Target speed 1m/s.
        '1': {
            'model': 'Hamner_modified',
            'targetSpeed': 1.00,
            'N': 25},
        # Weaker gluteus muscles (50% strength).
        '2': {
            'model': 'Hamner_modified_weakerGluts',
            'targetSpeed': 1.33,
            'N': 25},
        # Contact models with higher stiffness (10e6 instead of 1e6)
        '3': {
            'model': 'Hamner_modified_stifferContacts',
            'targetSpeed': 1.33,
            'N': 25},
        # Full gait cycle
        '4': {
            'model': 'Hamner_modified',
            'targetSpeed': 1.33,
            'N': 50,
            'gaitCycleSimulation': 'full'},
        # Full gait cycle with weaker right gluteus muscles (50% strength).
        '5': {
            'model': 'Hamner_modified_weakerRightGluts',
            'targetSpeed': 1.33,
            'N': 50,
            'gaitCycleSimulation': 'full'},
        }    
    
    return settings
