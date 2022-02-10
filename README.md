# BasicReservoirComputing
Implementation of a reservoir computer with functions to optimize the parameters and calculate the Lyapunov exponents

Please cite if using this code:
Platt, J. A., Penny, S. G., Smith, T. A., Chen, T.-C., & Abarbanel, H. D. I. (2022). A Systematic Exploration of Reservoir Computing for Forecasting Complex Spatiotemporal Dynamics. arXiv http://arxiv.org/abs/2201.08910

Examples are found under the examples folder in the github repo.

To install the package should be registered.
```Julia
import Pkg
Pkg.add("BasicReservoirComputing")
```

if not then add from github
```Julia
import Pkg
Pkg.add("git@github.com:japlatt/DynamicalRC.git")
```

## Basic Functionality
The code is built around the rc object

```Julia
struct rc 
    """Structure holding the rc parameters and matrixes
    
    Args:
        D::Int : Input system Dimension
        N::Int : Reservoir Dimension
        SR::Float : Spectral Radius
        ρA::Float : Density of A
        α::Float : Leak Rate
        σ::Float : Input Strength
        random_state::Int : Random seed
        σb::Float : Bias
        β::Float : Tikhonov regularization
        (params)::NamedTuple : Named tuple with fields SR, α, σ, σb, β
    """
    function rc(D, N, ρA, Δt, SR, α, σ, σb, β;
                 random_state=111111)
    end
    function rc(D, N, ρA, Δt, params::NamedTuple; random_state=111111)
        rng = MersenneTwister(random_state)
    end
end
```
To initialize the rc we can either specify all the parameters or pass in a few along with a named tuple which contains the fields (SR, α, σ, σb, β).  This functionality is mainly to help with integration with the optimization code

Once the rc has been initialized we can use the function train_RC
```Julia
function train_RC(rc::rc, u; spinup = 100, readout = false)
    """Train the rc

    This function sets the matrix rc.Wout
    u ∈ D×T
    r ∈ N×T

    Args:
        rc : reservoir computer
        u : Array of data of shape D×Time
        spinup : Amount of data to use for spinup
    Returns:
        if readout=true: Wout*r, the readout of the rc training data
        else: return the last state of r, can be used to predict
              forward from the training data 
    """
```
which takes in the rc and the training data u.  u is a matrix of equally spaced data of shape (D, Time).  Training will set the field Wout in the rc.

We can do a single forecast over nsteps.
```Julia
function forecast_RC(rc::rc, nsteps; uspin=nothing, r0 = nothing)
    """Forecast nsteps forward in time from the end of uspin or from rc state r0

    Make sure to train the RC before forecasting.  Requires either uspin or r0.
    If both are provided will use uspin to set r0.
    """
```
The functionality is that we can either give a reservoir state r0 to start the forecast or some data uspin in order to "spin up" the reservoir.  See the paper for more discussion.  If we'd like to test an ensemble off different test data then we can use test_RC() to return an array of valid prediction time.
```Julia
function test_RC(rc::rc, test_arr; ϵ=0.3, spinup=100)
    """Test the RC on all the forecasts in test_arr

    Args:
        rc : reservoir computer
        test_arr : array of forecasts DxT

    Returns:
        valid prediction time of all the tests
    """
```
See the paper for the formula as to how this is calculated.

## Optimization
In order for the RC to be useful we must find the correct parameters for the RC for the given problem.  Here we use an optimization routine, again see paper for details.
```Julia
opt_lower_bounds = @with_kw (SR_lb = 0.01, α_lb = 0.0, σ_lb = 1e-12, σb_lb = 0.0, β_lb = 1e-12)
opt_upper_bounds = @with_kw (SR_ub = 2.0, α_ub = 1.0, σ_ub = 1.0, σb_ub = 4.0, β_ub = 2.0)

struct opt
    """Optimization object

    Args:
        train_data, valid_data_list, spinup_steps, N, ρA
        lb::NamedTuple : use opt_lower_bounds to set lower bounds on params
        ub::NamedTuple : use opt_upped_bounds to set upper bounds on params
    """
```
The opt object takes in some DxT shaped training data, a list of DxT validation data and arguments for spinup steps, reservoir size/density and lower/upper bounds.  There are convenience functions for the lower and upper bounds which can be passed in with the changes specified e.g., lb = opt_lower_bounds(SR_lb = 0.5) and the rest set to default.

The optimization works using the algorithm CMAES <https://github.com/jbrea/CMAEvolutionStrategy.jl> but will hopefully be extended at some point to use any global optimization algorithm.

Now just call the algorithm
```Julia
function (f::opt)(;maxtime=200, popsize = 15, multithread = false)
    """Call future forecasting optimization routine

    This routine optimizes the rc by looking at a number of long term forecasts
    of the data and comparing them.
    """
```
so if optimization = opt(train_data, valid_data_list, spinup_steps, N, ρA) then just call optimization() to get the best parameters.  Multithreading can be a bit temperamental on some systems, make sure to initialize julia with multiple threads "julia -t2" if using this option.  maxtime and popsize may need to be adjusted depending on the difficulty of the problem.

An example from the examples folder is shown below
```Julia
function opt_rc(N, train_data, valid_data; Δt = 0.01)
    """Find the RC parameters through optimization"""
    # define some parameters
    nspin = 200
    ρA = 0.02
    D = size(train_data)[1]
    multithread = Threads.nthreads() > 1 ? true : false

    # Optimize the RC
    rc_opt = opt(train_data, valid_data, nspin, N, ρA,
                 lb = opt_lower_bounds(), ub = opt_upper_bounds())
    params, cost = rc_opt(maxtime=200, multithread=multithread)
    return params
end
```

## Lyapunov Exponents
One way to check the fidelity of the RC is to check that its Lyapunov exponents match those of the input data.  See the paper or 
>Jason A. Platt et al. "Robust forecasting using predictive generalized synchronization in reservoir computing". In: Chaos 31 (2021), p. 123118. URL: <https://doi.org/10.1063/5.0066013>.
for a more detailed description.

The routine here is 
```Julia
function Global_LEs(rc::rc, u, nspin; num_exp = 1, renorm_steps=10)
    """Find the Lyapunov Exponents of the rc

    Args:
        rc : reservoir computer
        u : input data, must be enough for the estimate.  Shorter=less accurate
        nspin : spinup steps to use
        num_exp : number of exponents to compute out of N
        renorm_steps : How many steps before renormalizing using QR decomposition
    Returns:
        Lyapunov Exponents : Array
    """
```
calculated with the help of DynamicalSystems.jl <https://juliadynamics.github.io/DynamicalSystems.jl/dev/>.  u is the data over which the estimate is done.  renorm_steps can help speed up the computation but will reduce accuracy if one is looking for the short term exponents.  Warning: the finite time LEs are not invariant over the attractor.  num_exp chooses the number of LEs to calculate.

We also expose the Tangent linear model TLM()/TLM!() in case one has need of it.  For instance in data assimilation algorithms.