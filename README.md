# MPS-BedMappingV1
This is the python repo for basal mapping from radar line data using multiple-point geostatistics (MPS). 

To run the Auto-BEL, the following dependencies must be met:
* [Jupyter](http://jupyter.org/) 
* [Python 3.8 or above](https://www.python.org/) 
* The above can be installed together via [Anaconda](https://www.anaconda.com/).

The following python library is required:
    Scikit learn >= 0.23
  
    Numpy >= 1.21.2
  
    Scipy >= 1.7.1
    
    Matplotlib >= 3.5.0

**Reference**: 

Yin, Z., Zuo, C., MacKie, E., & Caers, J. (in review). Mapping high-resolution basal topography of West Antarctica from radar data using non-stationary multiple-point geostatistics (MPS-BedMappingV1). 

Run the repo with Python Jupyter-Notebook by following steps - 
* [Step1. Training images evaluation and presentative patterns.ipynb](https://github.com/sdyinzhen/MPS-BedMappingV1/blob/main/Step1.%20Training%20images%20evaluation%20and%20presentative%20patterns.ipynb)
* [Step2. Estimate_Most_Probable_TIs_byPSO.ipynb](https://github.com/sdyinzhen/MPS-BedMappingV1/blob/main/Step2.%20Estimate_Most_Probable_TIs_byPSO.ipynb)
* [Step3. KDE_Estimation_of_P(TI|d).ipynb](https://github.com/sdyinzhen/MPS-BedMappingV1/blob/main/Step3.%20KDE_Estimation_of_P(TI%7Cd).ipynb)
* [Step4. Run_DS_simulation.ipynb](https://github.com/sdyinzhen/MPS-BedMappingV1/blob/main/Step4.%20Run_DS_simulation.ipynb)
