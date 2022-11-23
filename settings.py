def getSettings():
        
    settings = {        
        # Hamner model - 50 mesh intervals
        '0': {
            'model': 'Hamner_modified',
            'guessType': 'hotStart',
            'targetSpeed': 1.33,
            'N': 50,
            'modelMass': 62,
            'dampingMtp': 1.9},
        
        # Hamner model - 25 mesh intervals (faster)
        '1': {
            'model': 'Hamner_modified',
            'guessType': 'hotStart',
            'targetSpeed': 1.33,
            'N': 25,
            'modelMass': 62,
            'dampingMtp': 1.9}}    
    
    return settings
