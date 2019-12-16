import Pkg; Pkg.add("Plots"); Pkg.add("BenchmarkTools")
Pkg.add("DifferentialEquations")
Pkg.add("ParameterizedFunctions")
Pkg.add("StaticArrays")
#=f(u,p,t) = 1.01*u
u0=1/2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol = 1e-8)
using Plots
plot(sol, linewidth=5, title="Solution to the linear ODE with a thick line", xaxis="Time (t)", yaxis="u(t) (in μm)", label="My Thick Line!") # legend = false
plot!(sol.t, t->0.5*exp(1.01t), lw=3, ls=:dash, label="True Solution!")=#

#=
f(u,p,t) = 1/u
u0=3.0
tspan = (0.0, 1.0)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol = 1e-8, saveat=0.1 )
using Plots; gui()
plot(sol)
plot!(sol.t, t->(2*t+9)^.5, lw=3, ls=:dash, label="True Solution!")=#
#=
function lorenz!(du, u, p,t)
    σ, ρ,β = p
    du[1] =  σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3])-u[2]
    du[3] = u[1]*u[2]-β*u[3]
    end
u0=[1.0,0.0,0.0]
p=(10,28,8/3)
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob)
A=Array(sol)
using Plots; gr()
@btime plot(sol, vars=(0,1))
=#
#=
using ParameterizedFunctions
lv! = @ode_def LotkaVolterra begin
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
end a b c d
u0 = [1.0,1.0]
p = (1.5,1.0,3.0,1.0)
tspan = (0.0,10.0)
prob = ODEProblem(lv!,u0,tspan,p)
sol = solve(prob)
plot(sol)=# #predator prey model, and using the parimitzation stuff

#= using different internal types in particular matrices
A  = [1. 0  0 -5
      4 -2  4 -3
     -4  0  0  1
      5 -2  2  3]
u0 = rand(4,2)
tspan = (0.0,1.0)
f(u,p,t) = A*u
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)=##=
A  = @SMatrix [ 1.0  0.0 0.0 -5.0
                4.0 -2.0 4.0 -3.0
               -4.0  0.0 0.0  1.0
                5.0 -2.0 2.0  3.0]
u0 = @SMatrix rand(4,2)
tspan = (0.0,1.0)
f(u,p,t) = A*u
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)=#
using DifferentialEquations, ParameterizedFunctions
van! = @ode_def VanDerPol begin
  dy = μ*((1-x^2)*y - x)
  dx = 1*y
end μ
using BenchmarkTools

prob = ODEProblem(van!,[0.0,2.0],(0.0,6.3),1e6)
@btime solve(prob)
plot(sol, ylims = (-10.0, 10.0))
