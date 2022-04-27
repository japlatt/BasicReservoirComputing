using Plots
using Random
using DifferentialEquations
using DynamicalSystems

function make_data(system::DynamicalSystem;
                   train_time = 100, nspin=500, test_time = 15, n_test = 10, Δt=0.01)
    """make training and testing data from the dynamical system"""
    options = (alg = Vern9(), abstol = 1e-12,reltol=1e-12)
    D = length(system.u0)
    train_data = trajectory(system, train_time, system.u0.*rand(D).+rand(D);
                            Δt=Δt,
                            Ttr=nspin, diffeq=options)
    test_data = Matrix{Float64}[]
    for i in 1:n_test
        data = trajectory(system, test_time, system.u0.*rand(D).+rand(D);
                          Δt=Δt,
                          Ttr=nspin, diffeq=options)
        push!(test_data, collect(Matrix(data)'))
    end
    return collect(Matrix(train_data)'), test_data
end

function plot_prediction(time, upred, utrue)
    """Plotting function"""
    p = []
    D = size(upred)[1]
    for i in 1:D
        if i == 1
            px = plot(time, upred[1, :], label="RC")
            plot!(px, time, utrue[1, :], label="Truth")
        else
            px = plot(time, upred[i, :])
            plot!(px, time, utrue[i, :], legend=false)
        end
        ylabel!(px, "X$i")
        if i!=D xticks!(px, Int[]) end
        push!(p, px)
    end
    
    xlabel!(p[end], "Time")
    return plot(p..., layout=(D, 1), link=:x)
end