module BasicReservoirComputing
"""Basic implementation of a Reservoir computer

Author: Jason Platt
Email: 1763platt@gmail.com
"""

module RC

export rc, train_RC, test_RC, forecast_RC, TLM, Global_LEs

using SparseArrays
using LinearAlgebra
using Random
using ArnoldiMethod
using DynamicalSystems
using Parameters
using Distributions


if (Threads.nthreads() > 1) & !Sys.isapple()
    using MKL
end

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
    D::Int # Input System Dimension
    N::Int # Reservoir Dimension
    SR::Float64 # Spectral Radius of A
    ρA::Float64 # Density of A
    α::Float64 # Leak Rate
    σ::Float64 # Input Strength
    σb::Float64 # Input Bias
    β::Float64 # Tikhonov Regularization
    A::SparseMatrixCSC{Float64, Int64} # Adjacency Matrix
    Win::Array{Float64,2} # Input Matrix
    Wout::Array{Float64,2} # Output Matrix
    W::Array{Float64,2} # A + Win Wout
    rng::MersenneTwister # Random Number Generator
    climstd::Vector{Float64} # climatological standard deviation
    Δt::Float64 # Time step of the data/RC
    function rc(D, N, ρA, Δt, SR, α, σ, σb, β;
                 random_state=111111)
        @assert SR > 0
        # Random number generator
        rng = MersenneTwister(random_state)

        # Initialize the connectivity matrix/Win and allocate memory
        A, Win, Wout, W, climstd = initialize_weights(D, N, SR, ρA, σ, rng)
        new(D, N, SR, ρA, α, σ, σb, β, A, Win, Wout, W, rng, climstd, Δt)
    end
    function rc(D, N, ρA, Δt, params::NamedTuple; random_state=111111)
        rng = MersenneTwister(random_state)
        @unpack SR, α, σ, σb, β = params
        @assert SR > 0
        A, Win, Wout, W, climstd = initialize_weights(D, N, SR, ρA, σ, rng)
        new(D, N, SR, ρA, α, σ, σb, β, A, Win, Wout, W, rng, climstd, Δt)
    end
end

function initialize_weights(D, N, SR, ρA, σ, rng)
	"""Initialize the adjacency matrix, Win and allocate memory"""
    A = get_connection_matrix(N, ρA, SR, rng)
    Win = rand(rng, Uniform(-σ, σ), N, D)
    Wout = Array{Float64}(undef, (D, N)) 
    W = Array{Float64}(undef, (N, N))
    climstd = Vector{Float64}(undef, D)
    return A, Win, Wout, W, climstd
end

function get_connection_matrix(N, ρA, SR, rng)
	"""Initialize A and rescale maximal eigenvalue"""
    uni = Uniform(-1.0, 1.0)
    rf(rng, N) = rand(rng, uni, N)
    A = sprand(rng, N, N, ρA, rf)
    λs, _ = partialeigen(partialschur(A, nev=1, tol=1e-3, which=LM())[1])
    maxeig = abs(λs[1])
    A = A.*(SR/maxeig)
    return A
end

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
    @assert size(u)[1] == rc.D
    r = generate(rc, u)
    @views compute_Wout(rc, r[:, 1+spinup:end], u[:, 1+spinup:end])
    rc.W[:, :] .= rc.Win*rc.Wout .+ rc.A
    rc.climstd[:] .= vec(std(u, dims=2))
    if readout == true return rc.Wout*r end
    return r[:, end]
end

function forecast_RC(rc::rc, nsteps; uspin=nothing, r0 = nothing)
    """Forecast nsteps forward in time from the end of uspin or from rc state r0

    Make sure to train the RC before forecasting.  Requires either uspin or r0.
    If both are provided will use uspin to set r0.
    """
    if !isnothing(uspin)
    	@assert size(uspin)[1] == rc.D
        rspin = generate(rc, uspin)
        r0 = rspin[:, end]
    end
    @assert !isnothing(r0)

    rfc = Array{Float64}(undef, (rc.N, nsteps))
    rfc[:, 1] = r0
    
    for t in 1:nsteps-1
        @views @inbounds @fastmath auto_rc!(rfc[:, t+1], rc, rfc[:, t])
    end
    return rc.Wout*rfc
end

function test_RC(rc::rc, test_arr; ϵ=0.3, spinup=100)
    """Test the RC on all the forecasts in test_arr

    Args:
        rc : reservoir computer
        test_arr : array of forecasts DxT

    Returns:
        valid prediction time of all the tests
    """
    vpt = Float64[]
    for (i,test) in enumerate(test_arr)
    	@assert size(test)[1] == rc.D
        utest = test[:, spinup:end]
        nsteps = size(utest)[2]
        upred = forecast_RC(rc, nsteps, uspin = test[:, 1:spinup])

        diff = sqrt.(mean(((utest .- upred)./rc.climstd).^2, dims=1))

        thresh = vec(diff .> ϵ)
        idx = findfirst(thresh)
        if isnothing(idx) idx = length(thresh) end
        push!(vpt, idx*rc.Δt)
    end
    return vpt
end

function auto_rc!(rtp1, rc::rc, rt)
    rtp1[:] .= rc.α.*tanh.(rc.A*rt .+ rc.Win*rc.Wout*rt .+ rc.σb).+(1 .- rc.α).*rt
end

function driven_rc!(rtp1, rc::rc, rt, ut)
    rtp1[:] .= rc.α.*tanh.(rc.A*rt .+ rc.Win*ut .+ rc.σb).+(1 .- rc.α).*rt
end


function generate(rc::rc, u)
    """Generate hidden RC states on input data u"""
    T = size(u)[2]
    r = Array{Float64}(undef, (rc.N, T))
    r[:, 1] = zeros(rc.N)
    for t in 2:T
        @views @inbounds @fastmath driven_rc!(r[:, t], rc, r[:, t-1], u[:, t-1])
    end
    return r
end

function compute_Wout(rc::rc, r, u)
	"""Linear regression to find Wout"""
    try
        rc.Wout[:, :] .= (Symmetric(r*r'+rc.β*I)\(r*u'))'
    catch e
        println("Training Failed")
        println(e) 
    end
end

function TLM(rc, r)
    """The tangent linear model maps how perturbations
    to the model evolve in time.
    """
    t0 = rc.α.*(1 .- tanh.(rc.W*r .+ rc.σb).^2)
    Q = t0.*rc.W
    Q[diagind(Q)].+=(1 - rc.α)
    return Q
end

function TLM!(Q, rc, r)
    """The tangent linear model maps how perturbations
    to the model evolve in time.
    """
    t0 = rc.α.*(1 .- tanh.(rc.W*r .+ rc.σb).^2)
    Q[:, :] .= t0.*rc.W
    Q[diagind(Q)].+=(1 - rc.α)
end

function Global_LEs(rc::rc, u, nspin; num_exp = 1, renorm_steps=10, num_evals = 1000)
    """Find the Lyapunov Exponents of the rc

    Args:
        rc : reservoir computer
        u : input data, must be enough for the estimate. length(u)>renorm_steps*num_evals+nspin
        nspin : spinup steps to use
        num_exp : number of exponents to compute out of N
        renorm_steps : How many steps before renormalizing using QR decomposition
        num_evals : Number of renormalization steps to conduct
    Returns:
        Lyapunov Exponents : Array
    """
    function f!(xnew, x, p, t)
        driven_rc!(xnew, rc, x, @view(p[1][:, t+1]))
    end
    function fJac!(Jnew, x, p, t)
        TLM!(Jnew, rc, x)
    end
    DS = DiscreteDynamicalSystem(f!, zeros(rc.N), [u], fJac!)
    return lyapunovspectrum(DS, num_evals, num_exp, Δt = renorm_steps, Ttr = nspin)/rc.Δt
end
end

module Opt

export opt, opt_lower_bounds, opt_upper_bounds

using ..RC
using LinearAlgebra
using Parameters
using CMAEvolutionStrategy

if (Threads.nthreads() > 1) & !Sys.isapple()
    using MKL
end

opt_lower_bounds = @with_kw (SR_lb = 0.01, α_lb = 0.0, σ_lb = 1e-12, σb_lb = 0.0, β_lb = 1e-12)
opt_upper_bounds = @with_kw (SR_ub = 2.0, α_ub = 1.0, σ_ub = 1.0, σb_ub = 4.0, β_ub = 2.0)

struct opt
    """Optimization object

    Args:
        train_data, valid_data_list, spinup_steps, N, ρA
        lb::NamedTuple : use opt_lower_bounds to set lower bounds on params
        ub::NamedTuple : use opt_upped_bounds to set upper bounds on params
    """
    lb::Array{Float64, 1} # lower bounds
    ub::Array{Float64, 1} # upped bounds
    train_data::Array{Float64, 2} # Training data D×T
    valid_data_list::Array{AbstractArray{Float64,2},1} # List of validation data
    spinup_steps::Int64 # Number of spinup steps
    N::Int64 # rc size
    ρA::Float64 # density of A
    function opt(train_data, valid_data_list, spinup_steps, N, ρA;
                 lb = opt_lower_bounds(), ub = opt_upper_bounds())
        @unpack SR_lb, α_lb, σ_lb, σb_lb, β_lb = lb
        @unpack SR_ub, α_ub, σ_ub, σb_ub, β_ub = ub
        lb = [SR_lb, α_lb, log(σ_lb), σb_lb, log(β_lb)]
        ub = [SR_ub, α_ub, log(σ_ub), σb_ub, log(β_ub)]
        @assert lb < ub
        @assert ρA <= 1.0
        @assert ρA >= 0.0
        new(lb, ub, train_data, valid_data_list, spinup_steps, N, ρA)
    end
end

function (f::opt)(;maxtime=200, popsize = 15, multithread = false)
    """Call future forecasting optimization routine

    This routine optimizes the rc by looking at a number of long term forecasts
    of the data and comparing them.
    """
    if multithread
         @assert Threads.nthreads() > 1 
    end
    function Loss(data, spinup_steps, rc::rc)
        """Loss function over 1 forecast"""
        uspin = @view data[:, 1:spinup_steps]
        utrue = @view data[:, f.spinup_steps:end]
        Ttrue = size(utrue)[2]
        upred = forecast_RC(rc, Ttrue; uspin=uspin)
        err = upred.-utrue
        weight = exp.(-(1:Ttrue)./Ttrue)
        return norm(err.*weight')
    end

    function obj(θ)
        """Objective function to optimize"""
        θ = untransform(θ)
        SR, α, logσ, σb, logβ = θ
        D, _ = size(f.train_data)
        reservoir = rc(D, f.N, f.ρA, 1.0, SR, α, exp(logσ), σb, exp(logβ))
        train_RC(reservoir, f.train_data)
        n_valid = length(f.valid_data_list)
        loss = zeros(n_valid)
        for i in 1:n_valid
            @inbounds @views loss[i] = Loss(f.valid_data_list[i], f.spinup_steps, reservoir)
        end
        return sum(loss)
    end

    function untransform(θ::AbstractArray{Float64, 1})
        return f.lb + (f.ub - f.lb).*θ/10
    end
    
    function transform(θ::Array{Float64, 1})
        θ[:] .= clamp.(10*(θ - f.lb)./(f.ub - f.lb), 0.0, 10.0)
    end

    x0 = (f.ub + f.lb)/2
    x0 = transform(x0)
    obj(x0)
    result = minimize(obj, x0, 4;
                      lower = zeros(5),
                      upper = 10.0*ones(5),
                      popsize = popsize,
                      multi_threading = multithread,
                      verbosity = 1,
                      seed = rand(UInt),
                      maxtime = maxtime)

    xopt = xbest(result)
    xopt = untransform(xopt)
    xopt = (SR = xopt[1], α = xopt[2], σ = exp(xopt[3]), σb = xopt[4], β = exp(xopt[5]))
    return xopt, fbest(result)
end
end
using .RC
using .Opt
export rc, train_RC, test_RC, forecast_RC, TLM, Global_LEs, opt, opt_lower_bounds, opt_upper_bounds

end
