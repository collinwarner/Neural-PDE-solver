using OrdinaryDiffEq, Flux, DiffEqFlux
#sample random data to set up ODEproblem
tspan = (0.0f0,1.5f0)
x = param(randn(2))
inputs = 1

#Error with using tracked u with ROCK2 odesolver
function dudt(u::TrackedArray,pp,t)
    Flux.Tracker.collect(u)
end
#error occurs when ROCK2() is used works for Tsit5()
#run above
current_model = ODEProblem(dudt, x, tspan, inputs)
diffeq_adjoint(inputs, current_model, ROCK2(); u0 = x)
