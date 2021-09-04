# import necessary package
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings("ignore")



## DSsim_point_based_mult_TIs - the Python 3 code package for direction sampling(DS). This DS method is point based.
## Authors: Zuo Chen <chenzuo789@outlook.com> and David Zhen Yin <yinzhen@stanford.edu>
## Reference: 
## Zuo, et al. (2020). A Tree-Based Direct Sampling Method for Stochastic Surface and Subsurface Hydrological Modeling. Water Resources Research, 56(2).
import numpy as np

def Standardization_MinMaxScaler(LocalArea, GlobalDomain, Global_Height, Global_Width):
    'Perform standardization operation on hard data, to scale the hard data scale between 0 and 1'
    
    elevation_min = np.min(LocalArea[np.isfinite(LocalArea)])
    elevation_max = np.max(LocalArea[np.isfinite(LocalArea)])

    GlobalDomain_standard = (GlobalDomain - elevation_min)/(elevation_max - elevation_min)

    return GlobalDomain_standard,  elevation_max, elevation_min


def Standardization_MinMaxScaler_Reverse(GlobalDomain_sd, elevation_max, elevation_min):
    '''De-standardization'''

    GlobalDomain_reverse = (GlobalDomain_sd *(elevation_max - elevation_min))+elevation_min 
    
    return GlobalDomain_reverse



def DSsim_point_based_mult_TIs(SimulationGrid,
                               TIs,
                               DS_PatternRadius = 16,
                               DS_Neighbors = 30, 
                               DS_DistanceThreshold_factor = 0.05,
                               TI_SearchFraction = 0.2):
    '''
    This is the main function of point-based direct sampling. 
    Parameters - 
    SimulationGrid: simulation grid with hard data values, no-data area as np.nan, 2D array.
    TIs: Trainning images, 3D array, [TI_numbers, y_dim, x_dim].
    DS_SearchingRadius: searching radius to to obtain the data pattern. 
    DS_Neighbors: maximum neighborhood hard data points in data searching pattern.
    DS_DistanceThreshold_factor: DS_Threshold = DS_DistanceThreshold_factor*(max(SimulationGrid) - min(SimulationGrid))
    TI_SearchFraction: fractions of TI to search when computing the distance, searching path is randomized.
    '''
    
    # specify the basic information about the simulation area
    SG_height, SG_width =SimulationGrid.shape[0], SimulationGrid.shape[1]
    SimulationGrid_List = np.ndarray.tolist(SimulationGrid)

    SGmax = SimulationGrid[np.isfinite(SimulationGrid)].max()
    SGmin = SimulationGrid[np.isfinite(SimulationGrid)].min()
    
    DS_Threshold = DS_DistanceThreshold_factor*(SGmax-SGmin)

    # specify hard data pattern
    DS_SearchingRadius = DS_PatternRadius
    Collection_y_List, Collection_x_List = Specify_ConditioningDataSequence_Spiral(DS_SearchingRadius)
    # Assign TI_SearchFraction 
    DS_Fraction = TI_SearchFraction
        
    ### specify the simulation path ###
    x = np.arange(0, SG_width)
    y = np.arange(0, SG_height)
    xx, yy = np.meshgrid(x, y)

        # index location of each pixel 
    Simulation_path_x_List = np.ndarray.tolist(xx.ravel().astype(int))
    Simulation_path_y_List = np.ndarray.tolist(yy.ravel().astype(int))
        # define simulation path by index numbers
    Simulation_path = np.arange(len(Simulation_path_x_List))
 
        # find locations with hard data observations & first simulate locations with hard data
    Simulation_path_hard = np.arange(len(Simulation_path_x_List))
    count=0
    for i in Simulation_path:
        if  np.any(np.isfinite(SimulationGrid[Simulation_path_y_List[i]-2:Simulation_path_y_List[i]+3, 
                                              Simulation_path_x_List[i]-2:Simulation_path_x_List[i]+3])):

            Simulation_path_hard[[count, i]] = Simulation_path_hard[[i, count]] 
            count+=1
        # randomize the path that is without harddata
    np.random.shuffle(Simulation_path_hard[count:]) 
    
    TIs_checked = []
    for simulation_index in tqdm(Simulation_path_hard):

        center_y = Simulation_path_y_List[simulation_index]
        center_x = Simulation_path_x_List[simulation_index]

        element = SimulationGrid_List[center_y][center_x]
        # visit an unknown point
        if np.isfinite(element):
            continue
        Conditioning_pattern_List, Conditioning_y_List, Conditioning_x_List = \
                                Extract_DSPattern_From_SimulationDomain(SimulationDomain_List = SimulationGrid_List,
                                                                        SG_height = SimulationGrid.shape[0],
                                                                        SG_width = SimulationGrid.shape[1],
                                                                        Center_y = center_y,
                                                                        Center_x = center_x,
                                                                        NeighborsAmount = DS_Neighbors,
                                                                        Collection_y_List = Collection_y_List,
                                                                        Collection_x_List = Collection_x_List)
        if(len(Conditioning_pattern_List)==0):
        # if the program does not find any hard data
            itr = 1
            while(np.isnan(element)):
                
                current_best_TI_index  = np.random.randint(low=0,high=TIs.shape[0],size=1)[0]
                # ensure not to visit empty TIs
                if np.count_nonzero(np.isnan(TIs[current_best_TI_index]))< \
                                TIs[current_best_TI_index].shape[0]*TIs[current_best_TI_index].shape[1]/5:
                
                    current_best_y  = np.random.randint(low=0,high=TIs[current_best_TI_index].shape[0],size=1)[0]
                    current_best_x  = np.random.randint(low=0,high=TIs[current_best_TI_index].shape[1],size=1)[0]
                    element = TIs[current_best_TI_index, current_best_y, current_best_x]
                    
                if (np.isnan(element)) & (itr>500):
                    print(f' ***Warining*** element is always {element}. TI has too many empty areas! TI index-{current_best_TI_index}')
                itr += 1
        else:
             # perform the exhaustive searching program to find a comparable pattern
                
            itr = 1
            while(np.isnan(element)):
                
                current_best_y = np.random.randint(low=0,high=TIs.shape[1],size=1)[0]
                current_best_x = np.random.randint(low=0,high=TIs.shape[2],size=1)[0]
                current_best_TI_index = np.random.randint(low=0,high=TIs.shape[0],size=1)[0]
                
                current_best_distance = 9999999999.9
                for TI_index in range(TIs.shape[0]):
                    # skip TIs that have too many empty areas
                    if np.count_nonzero(np.isfinite(TIs[TI_index]))> \
                                TIs[TI_index].shape[0]*TIs[TI_index].shape[1]/4*3:

                        # find the optimal value 
                        training_y, training_x, training_distance = \
                        SearchingProgram_DS_Exhaustive_v2(TrainingImage= TIs[TI_index],
                                                          TI_height = TIs[TI_index].shape[0],
                                                          TI_width = TIs[TI_index].shape[1],
                                                          Conditioning_pattern = np.asarray(Conditioning_pattern_List),
                                                          Conditioning_y = np.asarray(Conditioning_y_List),
                                                          Conditioning_x = np.asarray(Conditioning_x_List),
                                                          Distance_Threshold = DS_Threshold,
                                                          Searching_Fraction = DS_Fraction)

        #                print(f'TI_index:{TI_index}')
                        if(training_distance <= DS_Threshold):
                            current_best_y = training_y
                            current_best_x = training_x
                            current_best_TI_index = TI_index
                            break

                        elif(training_distance < current_best_distance):
                            current_best_y = training_y
                            current_best_x = training_x
                            current_best_distance = training_distance
                            current_best_TI_index = TI_index

                element = TIs[current_best_TI_index, current_best_y, current_best_x]
                
                if (np.isnan(element)) & (itr>500):
                    print(f' ***Warining*** element is always {element}. TI has too many empty areas! TI index-{current_best_TI_index}')
                
                itr += 1
        element = TIs[current_best_TI_index, current_best_y, current_best_x]
        SimulationGrid_List[center_y][center_x] = element
        
        TIs_checked.append(current_best_TI_index)
    return np.asarray(SimulationGrid_List), TIs_checked
#     return np.asarray(SimulationGrid_List)

def Specify_ConditioningDataSequence_Spiral(SearchingRadius):
    '''specify the sequence to collect hard data'''
    Collection_y_List = []
    Collection_x_List = []
    Collection_distance_List = []
    
    for y in range(-SearchingRadius,SearchingRadius+1):
        for x in range(-SearchingRadius,SearchingRadius+1):
            Collection_y_List.append(y)
            Collection_x_List.append(x)
            Collection_distance_List.append(y**2+x**2)
            
    Collection_y = np.array(Collection_y_List,dtype=int)
    Collection_x = np.array(Collection_x_List,dtype=int)
    Collection_distance = np.array(Collection_distance_List)
    
    Collection_y = Collection_y[ np.argsort(Collection_distance) ]
    Collection_x = Collection_x[ np.argsort(Collection_distance) ]
    
    Collection_y = np.delete(arr=Collection_y,obj=0)
    Collection_x = np.delete(arr=Collection_x,obj=0)
    
    
    return Collection_y.tolist(), Collection_x.tolist()

def Extract_DSPattern_From_SimulationDomain(SimulationDomain_List,
                                            SG_height, SG_width,
                                            Center_y, Center_x,
                                            NeighborsAmount,
                                            Collection_y_List,
                                            Collection_x_List):
    '''extract a DS pattern from simulation grid'''
    Conditioning_pattern_List = []
    Conditioning_y_List = []
    Conditioning_x_List = []
    
    for relative_y, relative_x in zip(Collection_y_List, Collection_x_List):
        location_y = Center_y + relative_y
        location_x = Center_x + relative_x
        
        if(location_y<0 or location_y>= SG_height or location_x<0 or location_x>=SG_width):
            continue
        
        element = SimulationDomain_List[location_y][location_x]
#         print(element)
        # only collect the point with hard data values
        if(np.isfinite(element)):
            Conditioning_pattern_List.append(element)
            Conditioning_y_List.append(relative_y)
            Conditioning_x_List.append(relative_x)
            
        if(len(Conditioning_pattern_List)==NeighborsAmount):
            return Conditioning_pattern_List, Conditioning_y_List, Conditioning_x_List
        
    return Conditioning_pattern_List, Conditioning_y_List, Conditioning_x_List

def SearchingProgram_DS_Exhaustive_v2(TrainingImage,
                                      TI_height, TI_width,
                                      Conditioning_pattern,
                                      Conditioning_y,
                                      Conditioning_x,
                                      Distance_Threshold, Searching_Fraction):
    
    '''Conduct a random sampling program to find a compatible pattern'''
    Searching_Num = int(TI_height * TI_width * Searching_Fraction)

    # specify the searching path
    Searching_path = np.arange(start=0,stop=TI_height*TI_width,step=1,dtype=int)
    np.random.shuffle(Searching_path)
    Searching_path = Searching_path[:Searching_Num]
    Searching_path_y_List = np.ndarray.tolist(Searching_path // TI_width)
    Searching_path_x_List = np.ndarray.tolist(Searching_path % TI_width)

    mean_Value = np.mean(TrainingImage[np.isfinite(TrainingImage)])
    max_Value = np.max(TrainingImage[np.isfinite(TrainingImage)])
    
    current_best_y = 0
    current_best_x = 0
    current_best_distance = 999999999.9

    Conditioning_n_sqrt = np.sqrt(len(Conditioning_pattern))

#     Conditioning_Num = len(Conditioning_pattern)

    # test a point in the training image
    for i in range(Searching_Num):
        # create a training pattern
        center_y, center_x = Searching_path_y_List[i], Searching_path_x_List[i]
        
        if np.isnan(TrainingImage[center_y,center_x]):
            continue 
        # if Training_pattern has empty locations.Replace them by TI mean value
        Training_pattern = np.repeat(mean_Value, len(Conditioning_y))

        y_patterns = center_y +Conditioning_y
        x_patterns = center_x +Conditioning_x
        
        # making sure the pattern is within the TI
        patterns_in_TI = (x_patterns<TI_width) & (x_patterns>=0) & (y_patterns<TI_height) & (y_patterns>=0) 

        Training_pattern[patterns_in_TI] = TrainingImage[y_patterns[patterns_in_TI],x_patterns[patterns_in_TI]]
        
        # skip if more than 1/5 traning patterns are empty
        if np.count_nonzero(np.isnan(Training_pattern))>= len(Conditioning_y)//5:
            continue
        # if Training_pattern has empty locations.Replace them by TI max value
#         else:
        Training_pattern[np.isnan(Training_pattern)] = max_Value
        
        # calculate distance between two patterns
        temp = Training_pattern-Conditioning_pattern
        
        pattern_distance = np.linalg.norm(temp)/Conditioning_n_sqrt
        
        if(pattern_distance <= Distance_Threshold):
            return center_y, center_x, pattern_distance
#             break
        elif(pattern_distance < current_best_distance):
            current_best_y = center_y
            current_best_x = center_x
            current_best_distance = pattern_distance

    return current_best_y, current_best_x, current_best_distance
	

def DS_global(SimulationGrid, LineBloc_space, 
              CandidateTI, 
              TIAssignment_Matrix, 
              DS_Neighbors, DS_SearchingRadius, DS_DistanceThreshold,TI_SearchFraction,
              padding_radius, save_dir, start_block):
    
    '''
    This is te function to run DS for the global area. 
    @parameters
    SimulationGrid: global simulation area in grid and containing hard data, 2D array. 
    LineBloc_space: global simulation area with block index values, same values as simulation path. 
                    2D array, same size as SimulationGrid
                    Simulation path is from the first block value to the last block value. 
    CandidateTI: all the TIs. 3D array. [TI_total_number, y_dim, x_dim]
    TIAssignment_Matrix: maxtrix containing assinged TI numbers to each block area.
                    2D array, [Block_numbers, TI_number_for_each_block]
    DS_Neighbors, DS_SearchingRadius, DS_DistanceThreshold,TI_SearchFraction: standard DS parameters
    
    padding_radius: overlaping size between neibourghing blocks, when performing Local DS simulation. 
    save_dir: directory to save the output. strings
    
    '''
    start = time.time()
    
    SG_Height, SG_Width = SimulationGrid.shape
    
    # obtain simulation path from bloc index in space. 
    # Simulation path is from the first block value to the last block value. 
    SimulationPath = np.unique(LineBloc_space).astype(int)
    itr = 1
    for bloc_index in SimulationPath:
        if bloc_index < start_block:
            itr += 1
            continue 
        # Determine the location of the target block
        argwhere_Bloc = np.argwhere(LineBloc_space==bloc_index)
        Left_Top_y = argwhere_Bloc[:,0].min()
        Right_Bottom_y = argwhere_Bloc[:,0].max()
        Left_Top_x = argwhere_Bloc[:,1].min()
        Right_Bottom_x = argwhere_Bloc[:,1].max()
        corner_y_min = max(Left_Top_y-padding_radius,0)
        corner_y_max = min(Right_Bottom_y+padding_radius,SG_Height)
        corner_x_min = max(Left_Top_x-padding_radius,0)
        corner_x_max = min(Right_Bottom_x+padding_radius,SG_Width)
        LocalLines = SimulationGrid[corner_y_min:corner_y_max,corner_x_min:corner_x_max]

        # Standardize Linedata to scale the LocalLine 0-1. by (x-min(x))/(max(x)-min(x)). 
        # The standardization is on the global area, but using local max and min. 
        SimulationGrid, elevation_max, elevation_min = Standardization_MinMaxScaler(LocalArea = LocalLines, 
                                                                                    GlobalDomain = SimulationGrid, 
                                                                                    Global_Height = SG_Height, 
                                                                                    Global_Width = SG_Width)

        # select TI to the target block area
        TIs_Selected_index = TIAssignment_Matrix[bloc_index]
        TIs_Selected = CandidateTI[TIs_Selected_index]
        print(f'The indices of selected TIs are {TIs_Selected_index} for block {bloc_index}')
        
        # Check If the selected TIs are proper - not proper TIs will have too many empty areas.
        for i in TIAssignment_Matrix[bloc_index]:
            if np.count_nonzero(np.isnan(CandidateTI[i]))> \
                                CandidateTI[i].shape[0]*CandidateTI[i].shape[1]/3:
                print(f'***Warning*** - TI No. {i} of this block is not a proper TI. It will NOT be used because it has large empty areas!!!')
            
        # Run DS for the target local area
        LocalLines_sd = np.copy(SimulationGrid[corner_y_min:corner_y_max,corner_x_min:corner_x_max])
        local_DSsim, TIchecked = DSsim_point_based_mult_TIs(SimulationGrid = LocalLines_sd,
                                                            TIs = TIs_Selected,
                                                            DS_PatternRadius = DS_SearchingRadius,
                                                            DS_Neighbors = DS_Neighbors, 
                                                            DS_DistanceThreshold_factor = DS_DistanceThreshold,
                                                            TI_SearchFraction = TI_SearchFraction)

        SimulationGrid[corner_y_min:corner_y_max,corner_x_min:corner_x_max] = np.copy(local_DSsim)

        # Reverse the standardization to original scale
        SimulationGrid = Standardization_MinMaxScaler_Reverse(GlobalDomain_sd = SimulationGrid, 
                                                              elevation_max = elevation_max, 
                                                              elevation_min = elevation_min)
        
        # if itr%2 == 1:
        #     np.save(save_dir + '/SimulationGrid_stp'+str(itr), SimulationGrid)

        itr += 1
        plt.imshow(SimulationGrid,vmin=-3000,vmax=1500,cmap='terrain',interpolation='nearest')
        plt.colorbar(fraction=0.01)
        plt.title(f'Global Domain After De-standardization')
        plt.show()
        
    np.save(save_dir+'/SimulationGrid_final', SimulationGrid)
    
    end = time.time()
    time_total = np.round((end - start)/3600, 3)
    print(f'Runing time is {time_total} hrs')
    
    return SimulationGrid