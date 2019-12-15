using JLD2, Printf, Plots
using ApproxFun,OrdinaryDiffEq, Sundials
using LinearAlgebra
using Plots;# gr()
using Flux, DiffEqFlux, StaticArrays
using CuArrays
function extract_data()
    filename = "C:/Users/Collin/Documents/MIT/UROP/Rackauckas/mixed_layer_simulation_Q-100_dTdz0.010_tau0.00_profiles.jld2"
    les_data = jldopen(filename, "r")
    #print(keys(les_data))
    #print(keys(les_data["parameters"]))
    #everything but coriolis bouyancy and clousure
    #important one is
    #If I wanted z coordinates
    z = collect(les_data["grid"]["zC"])
    #@show z[1], z[end]+z[1]
    #read in temperature
    T = [] #temp
    t_first = [] #time
    Φ = [] #flux term (wT) for key purposes
    tmpT = les_data["timeseries"]["T"] #output the iteration number,

    for j in keys(tmpT)
        push!(T, les_data["timeseries"]["T"][j])
        push!(t_first, les_data["timeseries"]["t"][j])
        push!(Φ, les_data["timeseries"]["wT"][j])
    end
    #display(t)
    #
    #T_new = T[2:end-1]
    tmp_plot = []
    tmpT_plot = Matrix{Float64}(I, 256, 58)
    T_plot = []
    index_i = 1
    initial_u = T[1][1,1,2:end-1]
    #display(initial_u)
    #tmpT_plot[1,1] = 400
    #display(tmpT_plot[1,1])
    #Φ_plot = []
    #display(length(1:20:floor(Int, length(t_first) /1 * 1.0)))
    for i in 2:257
        index_j = 1
        for j in 1:20:floor(Int, length(t_first) /1 * 1.0)
            #display(tmpT_plot[i,index])
            #for j in 22:20:22#floor(Int, length(t) /1 * 1.0)

            #days = @sprintf("%.1f", t[j]/86400)

            tmpT_plot[index_i,index_j] = T[j][1,1,i]
            index_j += 1
            #Φ_plot =Φ[j][1,1,2:(end-1)]
            #push!(T_plot, tmp_plot)
            #p1 = plot(tmp_plot, z)
            #display(plot(Φ_plot, z))
            #display(plot(p2))
            #println(size(Φ_plot))


            #println(length(Φ[2]))
            #p1 = plot(tmp_plot, z)
            #p2 = plot(tmp_plot, z)

            #display(plot(p2))
        end
        index_i+=1
    end
    #display(tmpT_plot)
    #display(length(T[1]))
    #display(tmpT_plot)
    #t_new = @SVector t
    #t = @SVector [t_first[j] for j in 1:1153]

    return t_first, tmpT_plot, initial_u
end


function do_Ml(t, temp_data, initial_u)


    #set up variables

    datasize = 58 #number of timepoints in the interval
    N = 256 #number of steps in z in the interval
    tspan = (0.0f0,1.5f0) #start and end time with better precision
    tspan = (0.0f0, 691201.5042516432f0)
    #t = range(tspan[1], tspan[2], length = datasize) #time range

    #Finite Difference Method PDE -> ODE
    (z0, z_final) = (-0.1953125, -99.8046875)
    dz = (z_final+z0)/(N) #step size in z
    d = ones(N-2) #diagnol
    dl = ones(N-3) #super/lower diagonal
    zv = zeros(N-2) #zero diagonal used to extend D* for boundary condtions
    #D2 discritization of ∂_zz
    D2 = diagm(-1=>dl, 0=>-2*d, 1 => dl)
    κ = 0.5
    D2_B = hcat(zv, D2, zv) #add space for the boundary conditions space for "ghost nodes"
    #we only solve for the interior space steps
    D2_B[1,1] = D2_B[end, end] = 1

    D2_B = (κ/(dz^2)).*D2_B #add the constant κ as the equation requires and finish the discritization

    r = zeros(N-2)
    r = [0;r;0]

    #Boundary Conditons matrix QQ? need to figure this out

    Q= Matrix{Int64}(I, N-2, N-2)
    QQ= vcat(zeros(1,N-2), Q, zeros(1,N-2))

    u0 = initial_u[2:end-1]

    #display(size(temp_data[1]))
    training_data = Array(temp_data)
    #display(training_data)


    #input data, using above or the Temperature stuff

    ml_layers = Chain(Dense(N, 50, tanh), Dense(50, N-2))

    inputs = D2_B, QQ, ml_layers, r
    #display(Dense(N, 50, tanh))
    function dudt(u::TrackedArray,pp,t)
        D2_B,QQ,ml_layers,r = pp
        Flux.Tracker.collect((ml_layers(QQ * u)) + D2_B * (QQ * u))
    end


    function dudt(u::AbstractArray,inputs , t)
        D2_B, QQ, ml_layers, r = inputs
        #train the neural network on the D1*QQ*Φ term, don't need D1, just the size which is QQ*u
        Flux.data(layers(QQ*u)) + D2_B*(QQ*u)
    end

    current_model = ODEProblem(dudt, u0, tspan, inputs)

    param_u = param(u0)

    #Loss function
    function modeled_ode()
        diffeq_adjoint(inputs, current_model, Tsit5(), u0 = param_u, saveat = t)
    end

    loss_funct() = sum(abs2, training_data .- modeled_ode())

    #number of iterations to train
    training_time = 1
    learning_rate = 0.1

    iter= Iterators.repeated((), training_time)
    opt = ADAM(learning_rate) #optimization term
    starting_pts = zeros(datasize)


    function cb()
        display(loss_funct())
        plt2 = scatter(t, training_data[1,:], label = "data")
        for i = 1:datasize
            starting_pts[i] = Flux.data((modeled_ode().u)[i][1])
        end
        scatter!(plt2, t, Flux.data(starting_pts), label="prediction")
        display(plot(plt2))
        loss_funct() < 0.9 && Flux.stop()
    end

    cb()

    lyrs = Flux.params(ml_layers)
    Flux.train!(loss_funct, lyrs, iter, opt, cb = cb )
    #display(loss_funct())
end
t, temp_data, initial_u = extract_data()

do_Ml(t, temp_data, initial_u)
