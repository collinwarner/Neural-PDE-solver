using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqOperators,  LinearAlgebra,  Plots, Sundials, CuArrays, CUDAnative, BenchmarkTools, Profile

function Neural_PDE(datasize, N, tspan, t)

#Discretization USED
    #Finite Difference Method PDE -> ODE
    dz = 1/(N) #step size in z
    d = ones(N-2) #diagnol5
    dl = ones(N-3) #super/lower diagonal
    zv = zeros(N-2) #zero diagonal used to extend D* for boundary condtions

    #D1 first order discritization of ∂_z
    D1= diagm(-1 => -dl, 0 => d)
    D1_B = hcat(zv, D1, zv)
    D1_B[1,1] = -1 #
    #D1_B = cu((1/dz)*D1_B)
    D1_B = cu((1/dz)*D1_B)

    #D2 discritization of ∂_zz
    D2 = diagm(-1=>dl, 0=>-2*d, 1 => dl)
    κ = 0.05
    D2_B = hcat(zv, D2, zv) #add space for the boundary conditions space for "ghost nodes"
    #we only solve for the interior space steps
    D2_B[1,1] = D2_B[end, end] = 1

    D2_B = cu((κ/(dz^2)).*D2_B) #add the constant κ as the equation requires and finish the discritization

    #Boundary Conditons matrix QQ
    Q= Matrix{Int64}(I, N-2, N-2)
    QQ= cu(vcat(zeros(1,N-2), Q, zeros(1,N-2)))

#END OF Discretization parameters


#Initial Conditions
    zs = (1:N) * dz
    z = zs[2:N-1]
    f0 = z -> exp(-200*(z-0.75)^2)
    u0 = f0.(zs)
    x = u0[2:N-1] |> gpu

    firstp =  D1_B, D2_B, QQ

    #training_data = get_data(t, tspan, firstp, x)
    full_tspan = (0.0, 1.5)
    temp_train = get_data(t, full_tspan, firstp, x)
    training_data = cu(collect(temp_train(t)))




    u0 = param(x) #|>gpu
    ann = Chain(Dense(N,50,tanh), Dense(50,N-2)) |>gpu
    p1 = Flux.data(DiffEqFlux.destructure(ann))
    p2 = vec(D2_B)
    p3 = vec(QQ)
    p4 = param([p1;p2;p3])
    ps = Flux.params(p4,u0)

    function dudt_(u::TrackedArray,p,t)

        Flux.Tracker.collect(DiffEqFlux.restructure(ann, p[1:length(p1)])(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u) + reshape(p[length(p1)+1:length(p1)+length(p2)], size(D2_B))*(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u))


    end
    function dudt_(u::AbstractArray,p,t)

        Flux.data(DiffEqFlux.restructure(ann, p[1:length(p1)])(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u)) + reshape(p[length(p1)+1:length(p1)+length(p2)], size(D2_B))*(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u)

    end
    predict_adjoint()  =   diffeq_adjoint(p4,prob,Tsit5(),u0=u0, saveat = t,abstol=1e-6,reltol=1e-6)
    loss_adjoint()  = sum(abs2, training_data - predict_adjoint()) #super slow dev the package, watch chris's video, inside the layer do something

    prob = ODEProblem{false}(dudt_,u0,tspan,p4)
    epochs = Iterators.repeated((), 10)
    learning_rate = ADAM(0.1)
    lyrs = Flux.params(p4)
    new_tf = 0.0f0
    tolerance = 0.01
    #solve PDE in smaller time segments to reduce likelyhood of divergence
    for i in 1:3
        #get updated time
        new_tf += 0.5f0
        tspan = (0.0f0,new_tf) #start and end time with better precision
        t = range(tspan[1], tspan[2], length = datasize) #time range

        #get data of forward pass
        training_data = cu(collect(temp_train(t)))

        #solve the backpass
        prob = ODEProblem{false}(dudt_,u0,tspan,p4)
        Flux.train!(loss_adjoint, lyrs, epochs, learning_rate)

        while (loss_adjoint() > tolerance)
            Flux.train!(loss_adjoint, lyrs, epochs, learning_rate)
            display(loss_adjoint())
        end
        #while (loss_adjoint() > 0.01)
        #    Flux.train!(loss_adjoint, lyrs, data, opt)
        #end
        #display solution after each run
        #cb()
    end
    cur_pred = []


    cb = function ()
        display(loss_adjoint())
        cur_pred = collect(Flux.data(predict_adjoint()))
        pl = scatter(t,training_data[1,:],label="data", legend =:bottomright)
        scatter!(pl,t,cur_pred[1,:],label="prediction")
        display(plot(pl))
    end
    cb()

    return training_data, cur_pred

end


datasize = 10 #number of timepoints in the interval
N = 4  #number of steps in z in the interval
tspan = (0.0f0, 2.0f0) #start and end time with better precision
t = range(tspan[1], tspan[2], length = datasize) #time range

function ode_i(u, p, t)


    D1_B, D2_B, QQ = p
    Φ= cos.(sin.(u.^3) .+ sin.(cos.(u.^2)))

    return D1_B*(QQ*Φ) + D2_B*(QQ*u)
end

function get_data(t, tspan, firstp, x)

    generate_data = ODEProblem(ode_i, x,tspan, firstp)
    true_sol = solve(generate_data, Tsit5(), abstol = 1e-9, reltol = 1e-9)
    return true_sol
end


 #Neural_PDE(datasize, N, tspan, t)
@profile Neural_PDE(datasize, N, tspan, t)

Juno.profiler()


#cur_pred = collect(Flux.data(predict_adjoint()))
#pl = scatter(t,training_data[1,:],label="data", legend =:bottomright, title =  "Trained PDE Integrator \n Loss: 0.00975225  x = 0.25", xlabel = "Time", ylabel = "Temperature (C)")
#scatter!(pl,t,cur_pred[1,:],label="prediction")
#display(plot(pl))

#Perfomance results

#accuracy time with reltol and abstol,

#e-6
