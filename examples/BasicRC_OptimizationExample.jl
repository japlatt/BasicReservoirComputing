using LinearAlgebra
using DynamicalSystems
using Parameters
using Plots

include("../src/BasicReservoirComputing.jl")
using .BasicReservoirComputing

include("utilities.jl")

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

function test_rc(system::DynamicalSystem, params::NamedTuple; Δt = 0.01)
    """Test the rc on the given system with params"""
    D=length(system.u0)
    utrain, utest_arr = make_data(system; train_time = 100, test_time = 25, n_test = 100)

    @unpack N, SR, ρA, α, σ, β, σb, nspin = params

    # Build and Train RC
    reservoir = rc(D, N, ρA, Δt, SR, α, σ, σb, β)
    train_RC(reservoir, utrain, spinup=nspin)

    # Forecast RC
    utrue = utest_arr[1][:, nspin:end]
    nsteps = size(utrue)[2]
    upred = forecast_RC(reservoir, nsteps, uspin = utest_arr[1][:, 1:nspin])
    t = collect(0:nsteps-1)*Δt

    p1 = plot_prediction(t, upred, utrue)

    # Test on many points
    vpt = test_RC(reservoir, utest_arr; ϵ=0.3, spinup=nspin)
    p2 = histogram(vpt, legend=false, 
                   overwrite_figure=false,
                   show=false)
    xlabel!(p2, "Valid Prediction Time")
    pfinal = plot(p1, p2, layout = (1, 2), show=true)
    display(pfinal)
    return reservoir
end

# First make some data
system = Systems.lorenz()
Δt = 0.01
train_time=100
valid_time = 15
n_valid = 15
utrain, uvalid_arr = make_data(system; train_time = train_time, 
                               test_time = valid_time, n_test = n_valid,
                               Δt = Δt)

# Now find the best parameters
N=200
params = opt_rc(N, utrain, uvalid_arr)

# Test the RC with the optimized params
reservoir = test_rc(system, merge(params, (N=N, ρA=0.02, nspin=200)))

# See how the lyapunov exponents compare
renorm = 10 # speeds up the calculation by not doing QR renormalization every step
nspin = 200

max_steps = trunc(Int, (train_time/Δt- nspin)/renorm) # training steps-nspin
rc_LEs = Global_LEs(reservoir, utrain, nspin, 
                    num_exp = length(system.u0),
                    renorm_steps=renorm,
                    num_evals = max_steps)
sys_LEs = lyapunovspectrum(system, 10000)
println("rc_LEs = $rc_LEs")
println("system_LEs = $sys_LEs")