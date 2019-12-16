using ApproxFun,OrdinaryDiffEq, Sundials
using LinearAlgebra
using Plots; gr()
using Flux, DiffEqFlux, CuArrays

#CuArrays.allowscalar(false)
#set up variables

datasize = 30 #number of timepoints in the interval
N = 4 #number of steps in z in the interval
tspan = (0.0f0,1.5f0) #start and end time with better precision
t = range(tspan[1], tspan[2], length = datasize) #time range

#Finite Difference Method PDE -> ODE
dz = 1/(N) #step size in z
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

#D1 first order discritization of ∂_z
D1= diagm(-1 => -dl, 0 => d)
D1_B = hcat(zv, D1, zv)
D1_B[1,1] = -1 #
D1_B = (1/dz)*D1_B

#Boundary Conditons matrix QQ? need to figure this out

Q= Matrix{Int64}(I, N-2, N-2)
QQ= vcat(zeros(1,N-2), Q, zeros(1,N-2))

#Initial Conditions
zs = (1:N) * dz
z = zs[2:N-1]
f0 = z -> exp(-0.5*z)#exp(-200*(z-0.75)^2)
u0 = f0.(zs)
#plot(zs, u0)

#get true solution
u0 = u0[2:N-1]
r = zeros(N-2)
r = [0;r;0]
p =D1_B,D2_B,QQ,similar(r),r,similar(r)

#each ODE at time _i
function ode_i(du, u, p, t)
    #this will be in the form of du/dt = D2*u + D1* Φ (Φ will eventually be a NN, rn using input data)
    #du = D2_B*QQ*u + D1_B*QQ*Φ

    D1_B, D2_B, QQ, tmp, r, tmp2 = p
    #General fromat is D*QQ*u creates ghost nodes for boundary conditions first before multiplying
    #by the discritized derivative
    #first get D2*u
    #mul!(tmp, QQ, u)
    #mul!(du, D2_B, tmp)

    ##now create "data" for Φ
    Φ = cos.(sin.(u.^3) .+ sin.(cos.(u.^2)))
    ###now get D1*Φ
    #mul!(tmp, QQ, Φ)
    #du .= du + D1_B*tmp
    #or just write the equation this way, :)
    du .= D1_B*(QQ*Φ) + D2_B*(QQ*u)
end

generate_data = ODEProblem(ode_i, u0, (0.0, 1.5), p)
true_sol = solve(generate_data, Tsit5(), saveat=t)

#plt = plot(z, [true_sol(0.0)])
#plot!(z, [true_sol(0.6)])
#display(plt)

training_data = Array(true_sol)
#

#input data, using above or the Temperature stuff

ml_layers = Chain(Dense(N, 50, tanh), Dense(50, N-2))

x = u0
inputs = D2_B, QQ, ml_layers, r
#display(Dense(N, 50, tanh))
function dudt(u::TrackedArray,pp,t)
    D2_B,QQ,ml_layers,r = pp
    Flux.Tracker.collect((ml_layers(QQ * u)) + D2_B * (QQ * u))
end


function dudt(u::AbstractArray,inputs , t)
    D2_B, QQ, ml_layers, r = inputs
    #train the neural network on the D1*QQ*Φ term, don't need D1, just the size which is QQ*u
    Flux.data(ml_layers(QQ*u)) + D2_B*(QQ*u)
end

current_model = ODEProblem(dudt, u0, tspan, inputs)

#param_u = param(u0)

#Loss function
function modeled_ode()
    diffeq_adjoint(inputs, current_model, Tsit5(), u0 =x, saveat = t)
end

loss_funct() = sum(abs2, training_data .- cpu(modeled_ode()))

#number of iterations to train
training_time = 200
learning_rate = 0.1

iter= Iterators.repeated((), training_time)
opt = ADAM(learning_rate) #optimization term
starting_pts = zeros(datasize)


function cb()
    display(loss_funct())
    #plt2 = scatter(t, training_data[1,:], label = "data")
    #cul = modeled_ode()
    for i = 1:datasize
        #starting_pts .= Flux.data(cul[1,:])#(modeled_ode().u)[i][1])
        starting_pts[i] = Flux.data((modeled_ode())[i][1])

    end
    #scatter!(plt2, t, Flux.data(starting_pts), label="prediction")
    #display(plot(plt2))
    #loss_funct() < 0.0009 && Flux.stop()
    #reset!()
end

cb()

lyrs = Flux.params(ml_layers)
Flux.train!(loss_funct, lyrs, iter, opt, cb = cb )
