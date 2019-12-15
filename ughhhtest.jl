using ApproxFun,OrdinaryDiffEq, Sundials
using LinearAlgebra
using Plots; gr()
using Flux, DiffEqFlux
#set up
datasize = 30
N = 4
tspan = (0.0f0,1.5f0)
t = range(tspan[1],tspan[2],length=datasize)
#FDM to invert PDE -> ODE
dx = 1/(N)
d = ones(N-2) # diagonal
dl = ones(N-3) # super/lower diagonal
zv=zeros(N-2)
#D2
ϵ=0.05
D2=diagm(-1 => dl, 0 => -2*d, 1 => dl)
D2new=hcat(zv,D2,zv)
D2new[1,1]=D2new[end,end]=1
D2now=(ϵ/(dx^2)) .*D2new
#D1 first order upwind
ϵ2=0.5
D1=diagm(-1 => -dl, 0 => d)
D1new=hcat(zv,D1,zv)
D1new[1,1]=-1
D1now=(ϵ2)*(1/dx) .* D1new
#boundary condition matrix QQ
Q=Matrix{Int64}(I,N-2,N-2)
QQ=vcat(zeros(1,N-2),Q,zeros(1,N-2))
#Initial conditions
xs = (1:N) * dx
x = xs[2:N-1]
f0 = x -> exp(-200*(x-0.75)^2)
u0 = f0.(xs)
plot(xs,u0)
#get true_sol
u0=u0[2:N-1]
r=zeros(N-2)
r=[0;r;0]
p =D1now,D2now,QQ,similar(r),r
function bur(du,u,p,t)
    D1now,D2now,QQ,tmp,r = p
    D2now = D1now + D2now
    mul!(tmp,QQ,u)
    r = tmp .+ r
    mul!(du,D2now,r)
end
prob = ODEProblem(bur, u0, (0.0,1.5), p)
true_sol=solve(prob,Tsit5(),saveat=t)
plot(x,[true_sol(0.0)])
plot!(x,[true_sol(0.6)])
ode_data = Array(true_sol)

#ML D1
ann = Chain(Dense(N,50,tanh), Dense(50,N-2))

pp=D2now,QQ,ann,r
function dudt(u::TrackedArray,pp,t)
    D2now,QQ,ann,r = pp
    Flux.Tracker.collect((ann(QQ * u + r)) + D2now * (QQ * u + r))
end
function dudt(u::AbstractArray,pp,t)
    D2now,QQ,ann,r = pp
    Flux.data(ann(QQ * u + r)) + D2now * (QQ * u + r)
end

prob2 = ODEProblem(dudt,u0,tspan,pp)

_x=param(u0)
#loss function
function predict_n_ode()
  diffeq_adjoint(pp,prob2,Tsit5(),u0=_x, saveat=t)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())
#callback and train
data = Iterators.repeated((), 200)
opt = ADAM(0.1)
pre_pts=zeros(datasize)
cb = function ()
  display(loss_n_ode())
  pl = scatter(t,ode_data[1,:],label="data")
  for i= 1:datasize
     pre_pts[i]= Flux.data((predict_n_ode().u)[i][1])
  end
  scatter!(pl,t,Flux.data(pre_pts),label="prediction")
  display(plot(pl))
  #loss_n_ode() < 0.09 && Flux.stop()
end

cb()

ps = Flux.params(ann)
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
