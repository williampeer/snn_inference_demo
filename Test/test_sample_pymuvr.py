import pymuvr

# define two sets of observations for two cells
observations_1 = [[[1.0, 2.3], # 1st observation, 1st cell
                   [0.2, 2.5, 2.7]],            # 2nd cell
                  [[1.1, 1.2, 3.0], # 2nd observation
                   []],
                  [[5.0, 7.8],
                   [4.2, 6.0]]]
observations_2 = [[[0.9],
                   [0.7, 0.9, 3.3]],
                  [[0.3, 1.5, 2.4],
                   [2.5, 3.7]]]
# set parameters for the metric
cos = 0.1
tau = 1.0
# compute distances between all observations in set 1
# and those in set 2
print(pymuvr.dissimilarity_matrix(observations_1,
                            observations_2,
                            cos,
                            tau,
                            'distance'))
# array([[ 2.40281585,  1.92780957],
#        [ 2.76008964,  2.31230263],
#        [ 3.1322069 ,  3.17216524]])
# compute inner products
pymuvr.dissimilarity_matrix(observations_1,
                            observations_2,
                            cos,
                            tau,
                            'inner product')
# array([[ 4.30817654,  5.97348384],
#        [ 2.08532468,  3.85777053],
#        [ 0.59639918,  1.10721323]])
# compute all distances between observations in set 1
pymuvr.square_dissimilarity_matrix(observations_1,
                                   cos,
                                   tau,
                                   'distance')
# array([[ 0.        ,  2.6221159 ,  3.38230952],
#        [ 2.6221159 ,  0.        ,  3.10221811],
#        [ 3.38230952,  3.10221811,  0.        ]])
# compute inner products
pymuvr.square_dissimilarity_matrix(observations_1,
                                   cos,
                                   tau,
                                   'inner product')
# array([[ 8.04054275,  3.3022304 ,  0.62735459],
#        [ 3.3022304 ,  5.43940985,  0.23491838],
#        [ 0.62735459,  0.23491838,  4.6541841 ]])