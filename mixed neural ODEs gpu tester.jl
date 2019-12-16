using DiffEqFlux, Flux, OrdinaryDiffEq, CuArrays, Plots
using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqOperators, CuArrays, LinearAlgebra, CUDAnative
#x = Float32[2.; 0.]|>gpu






tspan = Float32.((0.0f0,25.0f0))
datasize = 30 #number of timepoints in the interval
N = 4 #number of steps in z in the interval
tspan = (0.0f0,1.5f0) #start and end time with better precision
t = range(tspan[1], tspan[2], length = datasize) #time range

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
D1_B = cu((1/dz)*D1_B)


#D2 discritization of ∂_zz
D2 = diagm(-1=>dl, 0=>-2*d, 1 => dl)
κ = 0.5
D2_B = hcat(zv, D2, zv) #add space for the boundary conditions space for "ghost nodes"
#we only solve for the interior space steps
D2_B[1,1] = D2_B[end, end] = 1
D2_B = cu((κ/(dz^2)).*D2_B) #add the constant κ as the equation requires and finish the discritization

#D2_B = cu((κ/(dz^2)).*D2_B) #add the constant κ as the equation requires and finish the discritization



#Boundary Conditons matrix QQ? need to figure this out

Q= Matrix{Int64}(I, N-2, N-2)
#QQ= cu(vcat(zeros(1,N-2), Q, zeros(1,N-2)))
QQ= cu(vcat(zeros(1,N-2), Q, zeros(1,N-2)))



#Initial Conditions
zs = (1:N) * dz
z = zs[2:N-1]
f0 = z -> exp(-0.5*z)#exp(-200*(z-0.75)^2)
u0 = f0.(zs)
#plot(zs, u0)

#get true solution

x = u0[2:N-1] #|> gpu
r = zeros(N-2)
#r = cu([0;r;0])
r = cu([0;r;0])

firstp =D1_B,D2_B,QQ,similar(r),r,similar(r)
#p =D2_B,QQ,similar(r),r,similar(r)

function ode_i(du, u, p, t)


    D1_B, D2_B, QQ, tmp, r, tmp2 = p
    #Φ = cos.(sin.(u.^3) .+ sin.(cos.(u.^2)))

    Φ = CUDAnative.cos.(CUDAnative.sin.(u.^3) .+ CUDAnative.sin.(CUDAnative.cos.(u.^2)))

    du .= D1_B*(QQ*Φ) + D2_B*(QQ*u)
end

generate_data = ODEProblem(ode_i, gpu(x), (0.0, 1.5), firstp)
true_sol = solve(generate_data, Tsit5(), saveat=t)

training_data = Array(true_sol)

## --- Reverse-mode AD ---

#tspan = (0.0f0,25.0f0)
#u0 = Tracker.param(x)|>gpu
#_x = u0
u0 = Tracker.param(x) |> gpu
ann = Chain(Dense(N,50,tanh), Dense(50,N-2)) |> gpu
#p = param(Float32[-2.0,1.1]) |> gpu
p1 = Flux.data(DiffEqFlux.destructure(ann))
p2 =  vec(D2_B)
p3 = vec(QQ)
p4 = param([p1;p2;p3])
#ps = Flux.params(p3, u0)
#DiffEqFlux.restructure(ann, p)
#inputs = D2_B, QQ, ann#param(vcat(vec(D2_B), vec(QQ), vec(p)))




#inputs = param(vcat(vec(D2_B), vec(QQ), vec(p)))

function dudt_( u::TrackedArray,p,t)
    #D2_B, QQ, ann = p
    #Flux.Tracker.collect(u)
    D2_B = reshape(p[353:360], (2, 4))
    QQ = reshape(p[361:368], (4, 2))
    #ml_layers = reshape(p[17:368], (352, 1))
    #ann1 = DiffEqFlux.restructure(ann, ml_layers)
    #du .= Flux.Tracker.collect(DiffEqFlux.restructure(ann, ml_layers)(QQ*u) + D2_B*(QQ*u))
    #D2_B = reshape(p[1:8], (2, 4))
    #QQ = reshape(p[9:16], (4, 2))

    ann1 = DiffEqFlux.restructure(ann, p[1:352])

    ##display(size(QQ))
    #display(size)
    #mul!(temp, QQ, u)
    #mul!(temp2, D2_B, temp)
    Flux.Tracker.collect((ann(QQ*u)) + D2_B*(QQ*u))

end
function dudt_( u::AbstractArray,p,t)
    #D2_B, QQ, ann = p
    #D2_B = reshape(p[1:8], (2, 4))
    #QQ = reshape(p[9:16], (4, 2))
    D2_B = reshape(p[353:360], (2, 4))
    QQ = reshape(p[361:368], (4, 2))
    #ml_layers = reshape(p[17:368], (352, 1))
    #ann1 = DiffEqFlux.restructure(ann, ml_layers)
    #du .= Flux.Tracker.collect(DiffEqFlux.restructure(ann, ml_layers)(QQ*u) + D2_B*(QQ*u))
    #D2_B = reshape(p[1:8], (2, 4))
    #QQ = reshape(p[9:16], (4, 2))

    ann1 = DiffEqFlux.restructure(ann, p[1:352])

    #ann1 = DiffEqFlux.restructure(ann, p[1:352])
    #D2_B = reshape(p[1:8], (2, 4))
    #QQ = reshape(p[9:16], (4, 2))
    #ml_layers = reshape(p[17:368], (352, 1))
    #ann1 = DiffEqFlux.restructure(ann, ml_layers)
    #D2_B = reshape(inputs[1:8], (2, 4))
    #QQ = reshape(inputs[9:16], (4, 2))
    #c = reshape(inputs[17:368], (352, 1))
    #ann1 = DiffEqFlux.restructure(ann, c)
    #du .= Flux.data(DiffEqFlux.restructure(ann, ml_layers)(QQ*u)) + D2_B*(QQ*u)
    #du .= Flux.data(u)#Flux.data(DiffEqFlux.restructure(ann, p)(u))
     Flux.data(ann(QQ*u)) + D2_B*(QQ*u)

end
#x0 = param(x)
prob = ODEProblem(dudt_,u0,tspan,p4)
function modeled_ode()
    diffeq_adjoint(p4,prob,Tsit5(), u0 = u0, saveat = t)
end


loss_funct() = sum(abs, training_data .- cpu(modeled_ode()))
#modeled_ode()
loss_funct()
#number of iterations to train
training_time = 200
learning_rate = 0.1

iter= Iterators.repeated((), training_time)
opt = ADAM(learning_rate) #optimization term
starting_pts = zeros(datasize)

#Flux.data((modeled_ode()[1,:]))



function cb()
    display(loss_funct())
    #plt2 = scatter(t, training_data[1,:], label = "data")
    #cul = modeled_ode()
    #for i = 1:datasize
    #    starting_pts[i] = Flux.data(cul[1,i])#(modeled_ode().u)[i][1])
    #end
    #scatter!(plt2, t, Flux.data(starting_pts), label="prediction")
    #display(plot(plt2))
    loss_funct() < 0.0009 && Flux.stop()
end


cb()
#
#
lyrs = Flux.params(ann)
Flux.train!(loss_funct, lyrs, iter, opt, cb = cb)



#plt = scatter(t, training_data[1,:], label = "data")
#Flux.data(modeled_ode())
#scatter!(plt, t, Flux.data(modeled_ode())[:, 1], label = "prediction")
