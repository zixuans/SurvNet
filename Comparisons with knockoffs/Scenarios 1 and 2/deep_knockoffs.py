import numpy as np
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import data
import parameters
from sklearn import preprocessing

for i in range(25):
    seed=i+1
    print('No.',seed,'\n')
    
    X=np.loadtxt('../../Model-X/Dataset1/data/X_64_'+str(seed)+'.txt')

    X=preprocessing.scale(X)
    X_train=X[:8000]
    n=X_train.shape[0]
    p=X_train.shape[1]

# Compute the empirical covariance matrix of the training data
    SigmaHat = np.cov(X_train, rowvar=False)

# Initialize generator of second-order knockoffs
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train,0), method="sdp")

# Measure pairwise second-order knockoff correlations 
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

    print('Average absolute pairwise correlation: %.3f.' %(np.mean(np.abs(corr_g))))

# Load the default hyperparameters for this model
#training_params = parameters.GetTrainingHyperParams('gmm')

# Set the parameters for training deep knockoffs
    pars = dict()
# Number of epochs
    pars['epochs'] = 1000
# Number of iterations over the full data per epoch
    pars['epoch_length'] = 100
# Data type, either "continuous" or "binary"
    pars['family'] = "continuous"
# Dimensions of the data
    pars['p'] = p
# Size of the test set
    pars['test_size']  = int(0.3*n)
# Batch size
    pars['batch_size'] = int(0.25*n)
# Learning rate
    pars['lr'] = 0.001
# When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
# Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = 1000
# Penalty for the MMD distance
    pars['GAMMA'] = 1.
# Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = 1.
# Decorrelation penalty hyperparameter
    pars['DELTA'] = 1.
# Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = corr_g
# Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = [1.,2.,4.,8.,16.,32.,64.,128.]

# Initialize the machine
    machine = KnockoffMachine(pars)

# Train the machine
    print("Fitting the knockoff machine...")
    machine.train(X_train)

# Generate deep knockoffs
    Xk_train_m = machine.generate(X_train)
    print("Size of the deep knockoff dataset: %d x %d." %(Xk_train_m.shape))

    np.savetxt('bigdata/Xk_64_'+str(seed)+'.txt',Xk_train_m)
