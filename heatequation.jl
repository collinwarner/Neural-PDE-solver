using Plots, BenchmarkTools
L = 0.1 #m, rod thickness
n = 10  #number of simulation nodes(points discretized)
T0 = 0   #intitial temperature of the rod
T1s = 20 #initial temperature of Left wall
T2s = 20 #initial temperature of right wall

dx = L/n #node thickness
alpha = 0.0001; #thermal diffusivity

t_final = 60; #s, simulation time
dt = 0.1 #fixed time step

x = dx/2:dx:L-dx/2

T = ones(n)*T0

dTdt = zeros(n)

t = 0:dt:t_final

function heat_eq!(T, dTdt, alpha, T2s, T1s, T0, n, dx, dt, L)
    #for j = 1:length(t)
        for i = 2:n-1
            dTdt[i] = alpha*(T[i+1]+T[i-1]-2*T[i])/dx^2
        end
        dTdt[1] = alpha*((T[2]+T1s-2*T[1])/dx^2)
        dTdt[n] = alpha*((T2s+T[n-1]-2T[n])/dx^2)
        T = T .+ dTdt.*dt

        display(plot(x,T, title = string("Heat equation rod at time: ") ,
                xlabel = "x", ylabel = "Temp", xlims = (0,L-dx/2), ylims = (0,20)))

    #end


end

heat_eq!(T, dTdt, alpha, T2s, T1s, T0, n, dx, dt, L)
