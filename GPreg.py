from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, ExpSineSquared, WhiteKernel
from sklearn.model_selection import GridSearchCV
import numpy.random

def main():
    ### Data setup
    SIZE = 200
    nodes = [0.3,0.9,1.8,4.5]
    # some non-uniformly distributed x
    X = numpy.random.randn(SIZE)/5 + numpy.random.choice(nodes,size=SIZE)
    Xr = X.reshape(-1,1)
    # non-uniformly noisy function
    y = numpy.cos(X*2) + numpy.random.randn(SIZE)/(1+abs(X)/2)
    yr = y.reshape(-1,1)
    # grid points for plotting
    x = numpy.arange(nodes[0]-1,nodes[-1]+1,0.05)
    xr = x.reshape(-1,1)

    ### GP setup and training
    # note that WhiteKernel (incorrectly) assumes uniform noise
    # ConstantKernel modifies the mean of the GP when added
    ### non-grid-search version
    #kernel = RBF() + WhiteKernel()
    #gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(Xr,yr)
    ### grid-search version (possibly safer against local optimum)
    grid = {
        "kernel" : [RBF(1.0) + WhiteKernel() + ConstantKernel(1.0),
                    RBF(1.0) + WhiteKernel() + ConstantKernel(0.5),
                    RBF(1.0) + WhiteKernel() + ConstantKernel(2.0),
                    RBF(2.0) + WhiteKernel() + ConstantKernel(1.0),
                    RBF(2.0) + WhiteKernel() + ConstantKernel(0.5),
                    RBF(2.0) + WhiteKernel() + ConstantKernel(2.0),
                    RBF(0.5) + WhiteKernel() + ConstantKernel(1.0),
                    RBF(0.5) + WhiteKernel() + ConstantKernel(0.5),
                    RBF(0.5) + WhiteKernel() + ConstantKernel(2.0),
                    RBF(4.0) + WhiteKernel() + ConstantKernel(1.0),
                    RBF(4.0) + WhiteKernel() + ConstantKernel(0.5),
                    RBF(4.0) + WhiteKernel() + ConstantKernel(2.0),
                    RBF(0.25) + WhiteKernel() + ConstantKernel(1.0),
                    RBF(0.25) + WhiteKernel() + ConstantKernel(0.5),
                    RBF(0.25) + WhiteKernel() + ConstantKernel(2.0)]
    }
    grid = {"kernel" : [RBF(1.0) + WhiteKernel()]}
    estimator = GridSearchCV(GaussianProcessRegressor(random_state=0), grid, cv=5)
    estimator.fit(Xr, yr)
    gpr = estimator.best_estimator_

    ### evaluation on observed data
    print(gpr.score(Xr, yr))
    ### prediction
    p,s = gpr.predict(xr, return_std=True)
    p = p.reshape(-1)
    
    ### visualization
    from matplotlib import pyplot
    pyplot.fill_between(x, p-s, p+s, color="green", alpha=0.2)
    pyplot.plot(X,y,".",label="obs")
    pyplot.plot(x,p,label="pred", color="green")
    pyplot.plot(x,numpy.cos(x*2),label="true")
    pyplot.legend()
    pyplot.show()



if __name__ == "__main__":
    main()
