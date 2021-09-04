from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import euclidean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn import preprocessing
from sklearn.manifold import MDS
def Select_TI_For_LocalArea_plot(Num_Selected_TI,
                                TargetBloc,
                                LineBloc_space,
                                TI_Assignment_Matrix,
                                Block_distances,
                                MDS_coordinates_2D,
                                MDS_coordinates_2D_List, 
                                kde_bw = 0.3,
                                plot = True):

    '''
    This function select several TI for a specified local area considering both hard data and global correlations
     with other blocks
     @Parameters
    Num_Selected_TI: Number of TIs assgined for each block, int. Fixed for all local block areas
    
    TargetBloc: Index location of the target block, int. 
                It indicates the index number of target block, same as the designed simulation path values. 
                
    LineBloc_space:Simluation Grid with all block indexes. 2D array. The index values is same as simulation path value. 
    
    TI_Assignment_Matrix: selected TI from Step2 based on hard data only.

    Block_distances: Distance matrix between the Indexed blocks.It is the blockwised distances calculated from Step 3.
                        2D array. [Total_Block_number, Total_Block_number]
    
    MDS_coordinates_2D: TI distance from MDS
    MDS_coordinates_2D_List: "MDS_coordinates_2D" in list format
    '''
    
    # the key parameter in kernel density estimator
    KDE_Bandwidth = kde_bw
    
    # initialize the transition probability vector
    # MDS_coordinates_2D.shape[0] is the number of candidate TI
    TransitionRatio = np.zeros((MDS_coordinates_2D.shape[0]))
    
    Aggregation_weight_sum = 0.0
    

    # derive aggregation weights from Block distance
    if np.any(Block_distances != 0): 
        Aggregation_weights = 1-Block_distances/Block_distances.max()
        
    # if the target loation is empty,  Block_distances == 0. 
    # Then get the Aggregation_weights by spatial distance between the blocks.
    elif np.all(Block_distances== 0):
        argwhere_TargetBloc = np.argwhere(LineBloc_space==TargetBloc)
        x_centr = (argwhere_TargetBloc[:,1].max() + argwhere_TargetBloc[:,1].min() )//2
        y_centr = (argwhere_TargetBloc[:,0].max() + argwhere_TargetBloc[:,0].min() )//2      
        Spatial_dist = []   
        for bloc_index in range(len(Block_distances)):
            argwhere_Bloc = np.argwhere(LineBloc_space==bloc_index)
            x_centr_Bloc = (argwhere_Bloc[:,1].max() + argwhere_Bloc[:,1].min() )//2
            y_centr_Bloc = (argwhere_Bloc[:,0].max() + argwhere_Bloc[:,0].min() )//2
            dist = np.sqrt( (x_centr-x_centr_Bloc)**2 + (y_centr-y_centr_Bloc)**2)
            Spatial_dist.append(dist)
        Aggregation_weights = 1-Spatial_dist/max(Spatial_dist)
        
    # visit each local area in the simulation domain
    for bloc_index in range(len(Block_distances)):
        ### Modified, if the location is empty  
        if np.all(Block_distances== 0): 
             # obtain the specified training image
            # test 0
            TI_Index = TI_Assignment_Matrix[bloc_index]
            
            Aggregation_weight = Aggregation_weights[bloc_index]
            
            Aggregation_weight_sum += Aggregation_weight

            KDE_instance = \
                KernelDensity(kernel='gaussian',bandwidth=KDE_Bandwidth).fit(MDS_coordinates_2D[TI_Index])
            
            TransitionProbability_oneInstance = np.exp(KDE_instance.score_samples(MDS_coordinates_2D_List))

            TransitionProbability_oneInstance = TransitionProbability_oneInstance / np.sum(TransitionProbability_oneInstance)

            TransitionProbability_oneInstance += 0.00000001

            TransitionRatio_oneInstance = np.log( TransitionProbability_oneInstance / (1-TransitionProbability_oneInstance) )

            # accumulate the transition probability
            TransitionRatio = TransitionRatio + Aggregation_weight * TransitionRatio_oneInstance
            
        ###! If the location is the block itself and not empty
            
        elif(np.any(Block_distances != 0) and bloc_index == TargetBloc):

            # obtain the specified training image
            # test ==
            TI_Index =  TI_Assignment_Matrix[bloc_index]

            # Weight = (1+Block_Density/smallest_distance)
#             Aggregation_weight = (1+Block_Density)/min(Block_distances[Block_distances>0])
            Aggregation_weight = 1
            
            Aggregation_weight_sum += Aggregation_weight

            # calculate transition probability
            KDE_instance = KernelDensity(kernel='gaussian',bandwidth=KDE_Bandwidth).fit(MDS_coordinates_2D[TI_Index])

            TransitionProbability_oneInstance = np.exp(KDE_instance.score_samples(MDS_coordinates_2D_List))

            TransitionProbability_oneInstance = TransitionProbability_oneInstance / np.sum(TransitionProbability_oneInstance)

            TransitionProbability_oneInstance[TransitionProbability_oneInstance<=0] =  0.00000001
            TransitionProbability_oneInstance[TransitionProbability_oneInstance>=1] =  1-0.00000001
        
            TransitionRatio_oneInstance = np.log( TransitionProbability_oneInstance / (1-TransitionProbability_oneInstance) )

            # accumulate the transition probability
            if plot: 
                TransitionRatio = TransitionRatio + Aggregation_weight * TransitionRatio_oneInstance
                plt.scatter(MDS_coordinates_2D[:, 0], MDS_coordinates_2D[:, 1], c=TransitionProbability_oneInstance, cmap='jet', s=36)
                plt.xlabel(str('MDS1'),fontsize='large')
                plt.ylabel(str('MDS2'),fontsize='large')
                plt.tick_params(direction='in',labelsize='large')
                plt.colorbar()
                plt.title(f'The TI selection probability for B{bloc_index}')
                plt.show()

        # Including all other areas with flight lines
        elif np.any(Block_distances != 0):             
            # obtain the specified training image
            # test all
            TI_Index = TI_Assignment_Matrix[bloc_index]
            
            # Modified calculate the importance / weight / influence Use distacne from Step 3
        
            Aggregation_weight = Aggregation_weights[bloc_index]
            
            Aggregation_weight = float(Aggregation_weight)
            Aggregation_weight_sum += Aggregation_weight

            # calculate transition probability
            KDE_instance = KernelDensity(kernel='gaussian',bandwidth=KDE_Bandwidth).fit(MDS_coordinates_2D[TI_Index])

            TransitionProbability_oneInstance = np.exp(KDE_instance.score_samples(MDS_coordinates_2D_List))

            TransitionProbability_oneInstance = TransitionProbability_oneInstance / np.sum(TransitionProbability_oneInstance)

#             TransitionProbability_oneInstance += 0.00000001
            TransitionProbability_oneInstance[TransitionProbability_oneInstance<=0] =  0.00000001
            TransitionProbability_oneInstance[TransitionProbability_oneInstance>=1] =  1-0.00000001
        
            TransitionRatio_oneInstance = np.log( TransitionProbability_oneInstance / (1-TransitionProbability_oneInstance) )

            # accumulate the transition probability
            TransitionRatio = TransitionRatio + Aggregation_weight * TransitionRatio_oneInstance
            if plot:
                plt.scatter(MDS_coordinates_2D[:, 0], MDS_coordinates_2D[:, 1], c=TransitionProbability_oneInstance, cmap='jet', s=36)
                plt.xlabel(str('MDS1'),fontsize='large')
                plt.ylabel(str('MDS2'),fontsize='large')
                plt.tick_params(direction='in',labelsize='large')
                plt.colorbar()
                plt.title(f'The TI selection probability for A{bloc_index}')
                plt.show()

#             plt.scatter(MDS_coordinates_2D[:, 0], MDS_coordinates_2D[:, 1], c=TransitionRatio_oneInstance, cmap='jet', s=36)
#             plt.xlabel(str('MDS1'),fontsize='large')
#             plt.ylabel(str('MDS2'),fontsize='large')
#             plt.tick_params(direction='in',labelsize='large')
#             plt.colorbar()
#             plt.title(f'The transition ratio for area y{index_y}, x{index_x}')
#             plt.show()

                
    # normalize the transition probability
#     TransitionRatio = TransitionRatio / Aggregation_weight_sum!!!
    Transition_TemporaryVariable = np.exp(TransitionRatio)
    
    TransitionProbability = Transition_TemporaryVariable / (1+Transition_TemporaryVariable)
    
#     print(np.sum(TransitionProbability))
    TransitionProbability[np.isnan(TransitionProbability)] = 0 
    
    TransitionProbability = TransitionProbability / np.sum(TransitionProbability)    
    
    pdf = np.copy(TransitionProbability)
    if plot:
        colormap_max = np.max(TransitionProbability)
        plt.scatter(MDS_coordinates_2D[:, 0], MDS_coordinates_2D[:, 1],
                    c=TransitionProbability,
                    vmin=0.0,vmax=colormap_max,
                    cmap='jet', s=36)
        plt.xlabel(str('MDS1'),fontsize='large')
        plt.ylabel(str('MDS2'),fontsize='large')
        plt.tick_params(direction='in',labelsize='large')
        plt.colorbar()
        plt.title('The accumulative transition probability')
        plt.show() 
        
    Candidate_TI_Index = [167]
    while Candidate_TI_Index[0]>150:
        Candidate_TI_Index = np.random.choice(a=MDS_coordinates_2D.shape[0],
                                              size=Num_Selected_TI,
                                              replace=False,
                                              p=TransitionProbability.tolist())

    print(f'the selected TI {Candidate_TI_Index}')

    if plot:
        plt.scatter(MDS_coordinates_2D[:, 0], MDS_coordinates_2D[:, 1], s=36)
        plt.scatter(MDS_coordinates_2D[Candidate_TI_Index, 0], MDS_coordinates_2D[Candidate_TI_Index, 1],
                    color='red', s=49, label='Selected TI')
        plt.xlabel(str('MDS1'),fontsize='large')
        plt.ylabel(str('MDS2'),fontsize='large')
        plt.tick_params(direction='in',labelsize='large')
        plt.legend(scatterpoints=1, loc='upper right', shadow=False,fontsize='large')
        plt.title('The selected TI')
    #     plt.colorbar()
        plt.show()
    
    ### Plot TI pdf.

        plt.figure(figsize=(10,5))

        argsort = pdf.argsort()[::-1]
    #     argsort = np.delete(argsort, np.argwhere(argsort[:10]>150)[:,0])

        plt.plot(pdf[argsort],'k.', markersize=15)
        plt.ylim(-0.01, pdf.max()*1.2 )

        plt.title('PDF vs TI - A'+str(TargetBloc))
        plt.ylabel('PDF', fontsize = 14)
        plt.xlabel('Training images rank (after transition)', fontsize = 15)
        plt.xticks(fontsize=12), plt.yticks(fontsize=12)
        plt.show()
    
    
    return Candidate_TI_Index, pdf, Aggregation_weights, TransitionRatio