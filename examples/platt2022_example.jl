using Plots
using Random
using DifferentialEquations
using LinearAlgebra
using DynamicalSystems
using Parameters
using BasicReservoirComputing

include("utilities.jl")

@inline @inbounds function colpitts_eom(x, p, t)
    α = p[1]; Γ = p[2]; q = p[3]; η = p[4]
    du1 = α*x[2]
    du2 = -Γ*(x[1] + x[3]) - q*x[2]
    du3 = η*(x[2] + 1 - exp(-x[1]))
    return SVector{3}(du1, du2, du3)
end

@inline @inbounds function colpitts_jac(x, p, t)
    α = p[1]; Γ = p[2]; q = p[3]; η = p[4]
    J = @SMatrix [0  α  0;
                 -Γ -q -Γ;
                 η*exp(-x[1]) η 0]
    return J
end

@inline @inbounds function CL63_eom(x, p, t)
    xe,ye,ze, xt,yt,zt, xo,yo,zo = x
    sigma, rho, beta, S, k1, k2, tau, c, ce, cz = p

    dxedt = sigma*(ye - xe) - ce*(S*xt + k1)
    dyedt = rho*xe - ye - xe*ze + ce*(S*yt + k1)
    dzedt = xe*ye - beta*ze

    # Tropical atmosphere system
    dxtdt = sigma*(yt - xt) - c*(S*xo + k2) - ce*(S*xe + k1)
    dytdt = rho*xt - yt - xt*zt + c*(S*yo + k2) + ce*(S*ye + k1)
    dztdt = xt*yt - beta*zt + cz*zo

    # Tropical ocean system
    dxodt = tau*sigma*(yo - xo) - c*(xt + k2)
    dyodt = tau*rho*xo - tau*yo - tau*S*xo*zo + c*(yt + k2)
    dzodt = tau*S*xo*yo - tau*beta*zo - cz*zt
    return SVector{9}(dxedt,dyedt,dzedt,
                      dxtdt,dytdt,dztdt,
                      dxodt,dyodt,dzodt)
end

@inline @inbounds function CL63_jac(x, p, t)
    xe,ye,ze, xt,yt,zt, xo,yo,zo = x
    σ, ρ, β, S, k1, k2, τ, c, ce, cz = p
    J = @SMatrix [-σ σ 0 -ce*S 0 0 0 0 0;
                  ρ-ze -1 -xe 0 ce*S 0 0 0 0;
                  ye xe -β 0 0 0 0 0 0;
                  -ce*S 0 0 -σ σ 0 -c*S 0 0;
                  0 ce*S 0 ρ-zt -1 -xt 0 c*S 0;
                  0 0 0 yt xt -β 0 0 cz;
                  0 0 0 -c 0 0 -τ*σ τ*σ 0;
                  0 0 0 0 c 0 τ*(σ-S*zo) -τ -τ*S*xo;
                  0 0 0 0 0 -cz τ*S*yo τ*S*xo -τ*β]
    return J
end

function test_rc(system::DynamicalSystem, params::NamedTuple; 
                 Δt = 0.01, train_time=100, test_time = 50, n_test = 100)
    D=length(system.u0)
    utrain, utest_arr = make_data(system; train_time = train_time,
                                  test_time = test_time, n_test = n_test)

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
                   overwrite_figure=false, bins = 15)
    xlabel!(p2, "Valid Prediction Time")
    pfinal = plot(p1, p2, layout = (1, 2), show=true)
    display(pfinal)
    return reservoir
end

CL63 = (σ = 0.007, α = 1.00, SR = 0.01, β = 2.201375e-08, σb = 0.66,
        N = 2000, ρA = 0.02, nspin = 200)
L63 = (σ = 0.084,α = 0.60, SR = 0.80, β = 8.493901e-08, σb = 1.60,
       N = 800, ρA = 0.02, nspin = 200)
L9610d = (σ = 0.005, α = 0.72, SR = 0.21, β = 7.640822e-09, σb = 1.47,
          N = 2000, ρA = 0.02, nspin = 200)
L965d = (σ = 0.060, α = 0.70, SR = 0.58, β = 6.332524e-09, σb = 1.59,
         N = 800, ρA = 0.02, nspin = 200)
colpitts = (σ = 0.100, α = 1.00, SR = 1.20, β = 1.000000e-08, σb = 2.00,
            N = 800, ρA = 0.02, nspin = 200)
rossler = (σ = 0.066, α = 0.47, SR = 0.50, β = 2.101845e-09, σb = 1.23,
           N = 800, ρA = 0.02, nspin = 200)

sys_L63 = Systems.lorenz()
sys_L965d = Systems.lorenz96(5, F=8.0)
sys_L9610d = Systems.lorenz96(10, F=8.0)
sys_rossler = Systems.roessler()
sys_colpitts = ContinuousDynamicalSystem(colpitts_eom,
                                         rand(3),
                                         [5.0, 0.08, 0.7, 6.3],
                                         colpitts_jac)
sys_CL63 = ContinuousDynamicalSystem(CL63_eom,
                                     [-3.1, -3.1, 20.7, -3.1, -3.1, 20.7, -3.1, -3.1, 20.7],
                                     [10, 28, 8/3., 1, 10, -11, 0.1, 1, 0.08, 1],
                                     CL63_jac)
# change arguments to test different systems
trained_reservoir = test_rc(sys_L63, L63, test_time = 25);