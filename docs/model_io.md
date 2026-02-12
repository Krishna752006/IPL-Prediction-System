# Model I/O Contract

## Inputs
batting_team  
bowling_team  
venue  
over  
ball  

## Outputs
Predicted runs  
Predicted wickets  

### Derived Metrics (not directly modeled):
- Run rate
- Winner inference

## Preprocessing
- OneHotEncoder for categorical
- StandardScaler for numeric
