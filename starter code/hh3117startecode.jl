using ApproxFun,OrdinaryDiffEq, Sundials
using LinearAlgebra
using Plots; gr()
using Flux, DiffEqFlux
#set up
datasize = 30
N = 10
tspan = (0.0f0,1.5f0)
t = range(tspan[1],tspan[2],length=datasize)

#FDM to invert PDE -> ODE
dx = 1/(N)
#D2
d = ones(N-2) # diagonal
dl = ones(N-3) # super/lower diagonal
ϵ=0.05
zv=zeros(N-2)
D2=diagm(-1 => dl, 0 => -2*d, 1 => dl)
D2new=hcat(zv,D2,zv)
D2new[1,1]=D2new[end,end]=1
D2now=(ϵ/(dx^2)) .*D2new
Q=Matrix{Int64}(I,N-2,N-2)
QQ=vcat(zeros(1,N-2),Q,zeros(1,N-2))
D2now=convert(Array{Float32},D2now)
QQ=convert(Array{Float32},QQ)
#D1 first order upwind
ϵ2=0.5
D1=diagm(-1 => -dl, 0 => d)
D1new=hcat(zv,D1,zv)
D1new[1,1]=-1
D1now=(ϵ2)*(1/dx) .* D1new
D1now=convert(Array{Float32},D1now)

#Initial conditions
xs = (1:N) * dx
x = xs[2:N-1]
f0 = x -> Float32(exp(-200*(x-0.75)^2))
u0 = f0.(xs)
plot(xs,u0)

#get true_sol as ode_data
u0=u0[2:N-1]
p =D1now,D2now,QQ
r=zeros(N-2)
r=convert(Vector{Float32},r)
r=[0;r;0]

function bur(du,u,p,t)
    D1now,D2now,QQ = p
    D2now = D1now + D2now
    du .= D2now * (QQ * u + r)
  end

prob = ODEProblem(bur, u0, (0.0,1.5), p)
true_sol=solve(prob,Tsit5(),saveat=t)
plot(x,[true_sol(0.0)])
plot!(x,[true_sol(0.6)])
ode_data = Array(solve(prob,Tsit5(),saveat=t))

#ML the whole DE
dudt = Chain(Dense(N-2,50,tanh),Dense(50,N-2))
function predict_n_ode()
  neural_ode(dudt,u0,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

data = Iterators.repeated((), 1000)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = Flux.data(predict_n_ode())
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
  loss_n_ode() < 0.5 && Flux.stop()
end

# Display the ODE with the initial parameter values.
cb()

ps = Flux.params(dudt)
@time Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
