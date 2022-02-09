using Plots
using Random
using DifferentialEquations
using LinearAlgebra
using DynamicalSystems
using Parameters

include("../src/BasicReservoirComputing.jl")
using .BasicReservoirComputing

include("utilities.jl")

function test_rc(system::DynamicalSystem, params::NamedTuple; Δt = 0.01)
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
                   overwrite_figure=false)
    xlabel!(p2, "Valid Prediction Time")
    pfinal = plot(p1, p2, layout = (1, 2), show=true)
    display(pfinal)
    return reservoir
end

params = (N=200, σ=0.084, α=0.6, SR=0.8, ρA=0.02, β=8.5e-8, σb=1.6, nspin=200)
system = Systems.lorenz()
test_rc(system, params);