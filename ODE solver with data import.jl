using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqOperators,  LinearAlgebra,  Plots
using JLD2, Printf, Plots

function extract_data()
    filename = "C:/Users/Collin/Documents/MIT/UROP/Rackauckas/mixed_layer_simulation_Q-100_dTdz0.010_tau0.00_profiles.jld2"
    les_data = jldopen(filename, "r")

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
    tmp_plot = []
    T_plot =randn(258, 30)
    #Φ_plot = []
    index = 1
    for j in 1:40:floor(Int, length(t_first) /1 * 1.0)
        jndex = 1

        #for j in 22:20:22#floor(Int, length(t) /1 * 1.0)
        for i in 1:258
        #days = @sprintf("%.1f", t[j]/86400)
            T_plot[jndex,index] = T[j][1,1,jndex]
            jndex +1
    end
    #push!(tmp_t, t_first[j])
    index+=1

    end
    display(index)

    return t_first, T_plot
end

#CuArrays.allowscalar(true)
#tspan = Float32.((0.0f0,25.0f0))
datasize = 30 #number of timepoints in the interval
N = 258 #number of steps in z in the interval
#tspan = (0.0f0,1.5f0) #start and end time with better precision
#t = range(tspan[1], tspan[2], length = datasize) #time range

#Finite Difference Method PDE -> ODE
dz = 1/(N) #step size in z
d = ones(N-2) #diagnol
dl = ones(N-3) #super/lower diagonal
zv = zeros(N-2) #zero diagonal used to extend D* for boundary condtions

#D1 first order discritization of ∂_z
D1= diagm(-1 => -dl, 0 => d)
D1_B = hcat(zv, D1, zv)
D1_B[1,1] = -1 #
#D1_B = cu((1/dz)*D1_B)
D1_B = (1/dz)*D1_B


#D2 discritization of ∂_zz
D2 = diagm(-1=>dl, 0=>-2*d, 1 => dl)
κ = 0.05
D2_B = hcat(zv, D2, zv) #add space for the boundary conditions space for "ghost nodes"
#we only solve for the interior space steps
D2_B[1,1] = D2_B[end, end] = 1



D2_B = (κ/(dz^2)).*D2_B #add the constant κ as the equation requires and finish the discritization

#D2_B = cu((κ/(dz^2)).*D2_B) #add the constant κ as the equation requires and finish the discritization



#Boundary Conditons matrix QQ? need to figure this out

Q= Matrix{Int64}(I, N-2, N-2)
#QQ= cu(vcat(zeros(1,N-2), Q, zeros(1,N-2)))
QQ= vcat(zeros(1,N-2), Q, zeros(1,N-2))
t_span, T_plot = extract_data()

training_data = Array(T_plot[2:N-1, :])



tspan = (t_span[1], t_span[end])
t = range(t_span[1], t_span[end], length = datasize) #time range

x = T_plot[:,1][2:end-1]
plot(t, training_data[1,:])
display("starting ML")
u0 = param(x) #|>gpu
#tspan = (0.0f0,25.0f0)

ann = Chain(Dense(N,50,tanh), Dense(50,N-2)) #|>gpu

p1 = Flux.data(DiffEqFlux.destructure(ann))
p2 = vec(D2_B)
p3 = vec(QQ)
p4 = param([p1;p2;p3])
ps = Flux.params(p4,u0)

function dudt_(du,u::TrackedArray,p,t)

    du .= Flux.Tracker.collect(DiffEqFlux.restructure(ann, p[1:352])(reshape(p[361:end], size(QQ))*u) + reshape(p[353:360], size(D2_B))*(reshape(p[361:end], size(QQ))*u))


end



function dudt_(du,u::AbstractArray,p,t)

    du .= Flux.data(DiffEqFlux.restructure(ann, p[1:352])(reshape(p[361:end], size(QQ))*u)) + reshape(p[353:360], size(D2_B))*(reshape(p[361:end], size(QQ))*u)


end
prob = ODEProblem(dudt_,u0,tspan,p4)
display("solving adjoint1")
diffeq_adjoint(p4,prob,Tsit5(),u0=u0)#,abstol=1e-8,reltol=1e-8)
display("done adjoint1")
function predict_adjoint()
  diffeq_adjoint(p4,prob,Tsit5(),u0=u0,saveat=t)#, abstol=1e-8,reltol=1e-12)
end
display("starting loss")
loss_adjoint() = sum(abs2, training_data .- predict_adjoint())
loss_adjoint()

display("ending loss")
data = Iterators.repeated((), 10)
opt = ADAM(0.1)

pre_pts=zeros(datasize)

#iterations = 0
cb = function ()
   display(loss_adjoint())
   #if iterations %10 == 0
   #display(loss_adjoint())
   #pl = scatter(t,true_sol[1,:],label="data")
   #for i= 1:datasize
   #   pre_pts[i]= Flux.data((predict_adjoint())[i][1])
   #end
   #scatter!(pl,t,Flux.data(pre_pts),label="prediction")
   #display(plot(pl))
end#loss_n_ode() < 0.09 && Flux.stop()


# Display the ODE with the current parameter values.
#cb()
display("start that mutha fucking mL")
lyrs = Flux.params(p4)
Flux.train!(loss_adjoint, lyrs, data, opt, cb = cb)

#cb()

cur_pred = Flux.data(predict_adjoint())
pl = scatter(t,training_data[1,:],label="data", legend =:bottomright)

scatter!(pl,t,cur_pred,label="prediction")
display(plot(pl))


#loss_n_ode() < 0.09 && Flux.stop()
