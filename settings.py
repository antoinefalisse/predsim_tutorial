def getSettings():
        
    settings = {        
        # Hamner model
        '0': {
            'model': 'Hamner_modified',
            'guessType': 'hotStart',
            'targetSpeed': 1.33,
            'N': 50,
            'modelMass': 62,
            'dampingMtp': 1.9},     
        
        '1': {
            'model': 'Hamner_modified',
            'guessType': 'hotStart',
            'targetSpeed': 1.33,
            'N': 5,
            'modelMass': 62,
            'dampingMtp': 1.9},
        
        '2': {
            'model': 'Hamner_modified',
            'guessType': 'hotStart',
            'targetSpeed': 1.33,
            'N': 10,
            'modelMass': 62,
            'dampingMtp': 1.9},
        }    
    
    return settings
