using JLD2, Printf, Plots
using ApproxFun,OrdinaryDiffEq, Sundials
using LinearAlgebra
using Plots;# gr()
using Flux, DiffEqFlux, StaticArrays
function extract_data()
    filename = "C:/Users/Collin/Documents/MIT/UROP/Rackauckas/mixed_layer_simulation_Q-100_dTdz0.010_tau0.00_profiles.jld2"
    les_data = jldopen(filename, "r")
    #print(keys(les_data))
    #print(keys(les_data["parameters"]))
    #everything but coriolis bouyancy and clousure
    #important one is
    #If I wanted z coordinates
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
    tmp_t =[]
    T_plot = []
    #Φ_plot = []
    i= 0.0
    for j in 1:floor(Int, length(t_first) /1 * 1.0)
        #for j in 22:20:22#floor(Int, length(t) /1 * 1.0)
        i +=1
        #days = @sprintf("%.1f", t[j]/86400)
        tmp_plot =T[j][1,1,2:(end -1)]
        Φ_plot =Φ[j][1,1,2:(end-1)]
        push!(T_plot, tmp_plot)
        push!(tmp_t, t_first[j])


        #println(length(Φ[2]))
        #p1 = plot(tmp_plot, z)
        #p2 = plot(tmp_plot, z)

        #display(plot(p2))
    end

    #display(i)
    display(T_plot[:,1])
    display(size(t_first))
    ##t_new = @SVector t
    ##t = @SVector [t_first[j] for j in 1:1153]
    #p1 = plot(t_first, T_plot[:,1])
    ##display(plot(Φ_plot, z))
    ##display(plot(p2))
    ##println(size(Φ_plot))
    #display(p1)
end

extract_data()
