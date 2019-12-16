using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqOperators,  LinearAlgebra,  Plots, Sundials

#manifest to tomal. 


#CuArrays.allowscalar(true)
tspan = Float32.((0.0f0,25.0f0))
datasize = 30 #number of timepoints in the interval
N = 4 #number of steps in z in the interval
tspan = (0.0f0,1.5f0) #start and end time with better precision
t = range(tspan[1], tspan[2], length = datasize) #time range
dt = (tspan[2]-tspan[1])/datasize
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



#Initial Conditions
zs = (1:N) * dz
z = zs[2:N-1]
f0 = z -> z#exp(-200*(z-0.75)^2)
u0 = f0.(zs)
#plot(zs, u0)

#get true solution

x = u0[2:N-1] #|> gpu
r = zeros(N-2)
#r = cu([0;r;0])
r = [0;r;0]

firstp =D1_B,D2_B,QQ,similar(r),r,similar(r)
#p =D2_B,QQ,similar(r),r,similar(r)

function ode_i(du, u, p, t)


    D1_B, D2_B, QQ, tmp, r, tmp2 = p
    Φ = cos.(sin.(u.^3) .+ sin.(cos.(u.^2)))

    #Φ = CUDAnative.cos.(CUDAnative.sin.(u.^3) .+ CUDAnative.sin.(CUDAnative.cos.(u.^2)))

    du .= D1_B*(QQ*Φ) + D2_B*(QQ*u)
end

generate_data = ODEProblem(ode_i, x,(0.0, 1.5), firstp)
true_sol = solve(generate_data, Tsit5(), saveat=t)#, dt = dt)

training_data = Array(true_sol)



u0 = param(x) #|>gpu
#tspan = (0.0f0,25.0f0)

ann = Chain(Dense(N,50,tanh), Dense(50,N-2)) #|>gpu

p1 = Flux.data(DiffEqFlux.destructure(ann))
p2 = vec(D2_B)
p3 = vec(QQ)
p4 = param([p1;p2;p3])
ps = Flux.params(p4,u0)

function dudt_(du,u::TrackedArray,p,t)

    du .= Flux.Tracker.collect(DiffEqFlux.restructure(ann, p[1:length(p1)])(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u) + reshape(p[length(p1)+1:length(p1)+length(p2)], size(D2_B))*(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u))


end
function dudt_(du,u::AbstractArray,p,t)

    du .= Flux.data(DiffEqFlux.restructure(ann, p[1:length(p1)])(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u)) + reshape(p[length(p1)+1:length(p1)+length(p2)], size(D2_B))*(reshape(p[length(p1)+1+length(p2):end], size(QQ))*u)

end
prob = ODEProblem(dudt_,u0,tspan,p4)
diffeq_adjoint(p4,prob,Tsit5(),u0=u0, saveat = t)#,abstol=1e-8,reltol=1e-8)

function predict_adjoint()
  diffeq_adjoint(p4, prob, Tsit5(), u0=u0,saveat=t)#, abstol=1e-8,reltol=1e-12)
end
loss_adjoint() = sum(abs2, training_data .- predict_adjoint()) #super slow dev the package, watch chris's video, inside the layer do something
loss_adjoint()

data = Iterators.repeated((), 10)
opt = ADAM(0.1)

pre_pts=zeros(datasize)

#iterations = 0
cb = function ()
  display(loss_adjoint())
#   #if iterations %10 == 0
#   #display(loss_adjoint())
#   #pl = scatter(t,true_sol[1,:],label="data")
#   #for i= 1:datasize
#   #   pre_pts[i]= Flux.data((predict_adjoint())[i][1])
#   #end
#   #scatter!(pl,t,Flux.data(pre_pts),label="prediction")
#   #display(plot(pl))
end#loss_n_ode() < 0.09 && Flux.stop()


# Display the ODE with the current parameter values.
#cb()

lyrs = Flux.params(p4)
Flux.train!(loss_adjoint, lyrs, data, opt)#, cb = cb)

cb()

cur_pred = Flux.data(predict_adjoint())
pl = scatter(t,true_sol[1,:],label="data", legend =:bottomright)

scatter!(pl,t,cur_pred[1,:],label="prediction")
display(plot(pl))


#loss_n_ode() < 0.09 && Flux.stop()
