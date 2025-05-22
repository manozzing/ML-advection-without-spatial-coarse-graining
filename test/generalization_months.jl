import Pkg
Pkg.activate("../.")

using EarthSciData, EarthSciMLBase, GasChem, 
    DomainSets, ModelingToolkit, DifferentialEquations, Dates
using Plots, Statistics, GeoDataFrames, NCDatasets, StatsBase
using Flux, BenchmarkTools
using BSON: @save
using BSON: @load
using Base.Threads, LoopVectorization, Tullio, Random
using SimpleChains
using DelimitedFiles
using Plots.PlotMeasures

include("../src/advection_operators.jl")
include("../src/Flux_to_SC_params.jl")
include("../src/visualization_utils.jl")

## Set the time domain and the spatiotemporal resolution
days = 10
numsteps = days*24*12
space_factor = 1
temp_factor = 32
starting_month = 1
vertical_level = 1.0

## You will confirm the coarse-graining factor in both space and time.
println("space factor: "*string(space_factor)*"   temp factor: "*string(temp_factor))

### Single run for future use
ModelingToolkit.check_units(eqs...) = nothing; start = Dates.datetime2unix(Dates.DateTime(2018, starting_month, 1))
@parameters t
lons = collect(-130.0:0.3125*space_factor:-60.0); lats = collect(9.75:0.25*space_factor:60.0); levs = collect(vertical_level:1:vertical_level)
nsims = length(lats)*length(lons)*length(levs)
ntimesteps = numsteps÷temp_factor+1; s_per_timestep = 300*temp_factor; finish = start + s_per_timestep
model_ode = SuperFast(t) + FastJX(t); sys = structural_simplify(get_mtk(model_ode));
prob = ODEProblem(sys, [], (start, finish), []); sol = solve(prob, TRBDF2(), saveat=1800.0)
geos = GEOSFP{Float64}("0.25x0.3125_NA", t; coord_defaults = Dict(:lat => 34.0, :lon=>-100.0, :lev => 1));
latgrid = reshape(repeat(lats, outer = length(lons) * length(levs)), length(lats), length(lons), length(levs))
longrid = reshape(repeat(lons, outer = length(levs), inner = length(lats)), length(lats), length(lons), length(levs))
levgrid = reshape(repeat(levs, inner = length(lats) * length(lons)), length(lats), length(lons), length(levs))
defaults = ModelingToolkit.get_defaults(sys); vars = states(sys); ps = parameters(sys); indexof(sym, syms) = findfirst(isequal(sym), syms);

# Advection
Re = 6378000; vert_pressure_hpa = [1013.250]; Δy = Re*(lats[2] - lats[1])*π/180; fs = geos.filesets["A3dyn"]; default_time = Dates.DateTime(2018, starting_month, 1)
u_itp = EarthSciData.DataSetInterpolator{Float64}(fs, "U", default_time);
v_itp = EarthSciData.DataSetInterpolator{Float64}(fs, "V", default_time); 
start = Dates.datetime2unix(Dates.DateTime(2018, starting_month, 1)); h = 1
u_wind=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs)); v_wind=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs))

times = []
for h in 1:ntimesteps
    push!(times, start);
    for i in 1:length(lons)
        for j in 1:length(lats)
            u_wind[h,j,i,1] = Float32.(EarthSciData.interp!(u_itp, start, lons[i], lats[j], vertical_level));
            v_wind[h,j,i,1] = Float32.(EarthSciData.interp!(v_itp, start, lons[i], lats[j], vertical_level));
        end
    end
    start += s_per_timestep
    finish += s_per_timestep
end
push!(times, finish);

## Data loading, but you will need to download those 2-D datasets from the separate data documentation server and then upload in your local or cluster directory.
ds_2D = NCDataset("../data/2-D_generalization_altitudes/2D_test_nested_"*string(vertical_level)*"lev_"*string(days)*"day_"*string(starting_month)*"month_2018_"*string(space_factor)*"x"*string(temp_factor)*"t_+1.nc")
u_reference = zeros(Float32, ntimesteps, length(lats), length(lons), 1, 1); u_reference .= ds_2D["u"]

model, MP = Flux_model_params(space_factor, temp_factor)
loss(x, y) = Flux.Losses.mae(model(x), y); ps = Flux.params(model);

u_learned = zeros(Float32, ntimesteps, length(lats), length(lons), length(levs), 1)
for j in length(lats)÷3:length(lats)÷3*2
    for i in length(lons)÷3:length(lons)÷3*2
        u_learned[1,i,j,1,1] = 100
    end
end

@load "../data/model_params/CNN_trained_"*string(temp_factor)*"dt.bson_MODEL" model
ps = Flux.params(model);
model_SC, p_SC = param_Flux_to_SC(temp_factor, ps)

@time u_learned = pgm_ml_SC_2D(u_learned, ntimesteps, u_wind, v_wind, model_SC, p_SC, MP, length(lons), length(lats), temp_factor);
mae_2D_total = StatsBase.L1dist(reshape(u_learned[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)), reshape(u_reference[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)))/(ntimesteps*length(lats)*length(lons))
rmse_2D_total = StatsBase.L2dist(reshape(u_learned[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)), reshape(u_reference[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)))/sqrt(ntimesteps*length(lats)*length(lons))
r²_2D_total = Statistics.cor(reshape(u_learned[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)), reshape(u_reference[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)))^2
println("month: ", starting_month, "level: ", vertical_level, "temp_factor: ", temp_factor)
println(mae_2D_total, "    ", rmse_2D_total, "    ", r²_2D_total)





### Real game! Multiple run

Vertical_level = 1.0
for Temp_factor in [4, 8, 12, 16, 32]
    # Model parameters loading
    model, MP = Flux_model_params(space_factor, Temp_factor)
    loss(x, y) = Flux.Losses.mae(model(x), y); ps = Flux.params(model);
    @load "../data/model_params/CNN_trained_"*string(temp_factor)*"dt.bson_MODEL" model
    ps = Flux.params(model);
    model_SC, p_SC = param_Flux_to_SC(temp_factor, ps)

    for Starting_month in 1:12
        ModelingToolkit.check_units(eqs...) = nothing; start = Dates.datetime2unix(Dates.DateTime(2018, Starting_month, 1))
        @parameters t
        lons = collect(-130.0:0.3125*space_factor:-60.0); lats = collect(9.75:0.25*space_factor:60.0); levs = collect(Vertical_level:1:Vertical_level)
        nsims = length(lats)*length(lons)*length(levs)
        ntimesteps = numsteps÷Temp_factor+1; s_per_timestep = 300*Temp_factor; finish = start + s_per_timestep
        model_ode = SuperFast(t) + FastJX(t); sys = structural_simplify(get_mtk(model_ode));
        prob = ODEProblem(sys, [], (start, finish), []); sol = solve(prob, TRBDF2(), saveat=1800.0)
        geos = GEOSFP{Float64}("0.25x0.3125_NA", t; coord_defaults = Dict(:lat => 34.0, :lon=>-100.0, :lev => 1));
        latgrid = reshape(repeat(lats, outer = length(lons) * length(levs)), length(lats), length(lons), length(levs))
        longrid = reshape(repeat(lons, outer = length(levs), inner = length(lats)), length(lats), length(lons), length(levs))
        levgrid = reshape(repeat(levs, inner = length(lats) * length(lons)), length(lats), length(lons), length(levs))
        defaults = ModelingToolkit.get_defaults(sys); vars = states(sys); ps = parameters(sys); indexof(sym, syms) = findfirst(isequal(sym), syms);

        # Advection
        Re = 6378000; vert_pressure_hpa = [1013.250]; Δy = Re*(lats[2] - lats[1])*π/180; fs = geos.filesets["A3dyn"]; default_time = Dates.DateTime(2018, Starting_month, 1)
        u_itp = EarthSciData.DataSetInterpolator{Float64}(fs, "U", default_time);
        v_itp = EarthSciData.DataSetInterpolator{Float64}(fs, "V", default_time); 
        start = Dates.datetime2unix(Dates.DateTime(2018, Starting_month, 1)); h = 1
        u_wind=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs)); v_wind=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs))

        times = []
        for h in 1:ntimesteps
            push!(times, start);
            for i in 1:length(lons)
                for j in 1:length(lats)
                    u_wind[h,j,i,1] = Float32.(EarthSciData.interp!(u_itp, start, lons[i], lats[j], Vertical_level));
                    v_wind[h,j,i,1] = Float32.(EarthSciData.interp!(v_itp, start, lons[i], lats[j], Vertical_level));
                end
            end
            start += s_per_timestep
            finish += s_per_timestep
        end
        push!(times, finish);

        ## Data loading, but you will need to download those 2-D datasets from the separate data documentation server and then upload in your local or cluster directory.
        ds_2D = NCDataset("../data/2-D_generalization_altitudes/2D_test_nested_"*string(vertical_level)*"lev_"*string(days)*"day_"*string(starting_month)*"month_2018_"*string(space_factor)*"x"*string(temp_factor)*"t_+1.nc")
        u_reference = zeros(Float32, ntimesteps, length(lats), length(lons), 1, 1); u_reference .= ds_2D["u"]

        u_learned = zeros(Float32, ntimesteps, length(lats), length(lons), length(levs), 1)
        for j in length(lats)÷3:length(lats)÷3*2
            for i in length(lons)÷3:length(lons)÷3*2
                u_learned[1,i,j,1,1] = 100
            end
        end

        u_learned = pgm_ml_SC_2D(u_learned, ntimesteps, u_wind, v_wind, model_SC, p_SC, MP, length(lons), length(lats), Temp_factor);
        mae_2D_total = StatsBase.L1dist(reshape(u_learned[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)), reshape(u_reference[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)))/(ntimesteps*length(lats)*length(lons))
        rmse_2D_total = StatsBase.L2dist(reshape(u_learned[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)), reshape(u_reference[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)))/sqrt(ntimesteps*length(lats)*length(lons))
        r²_2D_total = Statistics.cor(reshape(u_learned[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)), reshape(u_reference[1:ntimesteps,:,:,:,:], ntimesteps*length(lats)*length(lons)))^2
        println("month: ", Starting_month, ". level: ", Vertical_level, ". temp_factor: ", Temp_factor)
        println(mae_2D_total, "    ", rmse_2D_total, "    ", r²_2D_total)
    end
end


## Visualization
r2 = readdlm("gen_season_r2.txt")
plot(r2[:,1], r2[:,2],
    xlabel="Month", ylabel="R²", xlabelfontsize=14, ylabelfontsize=14, 
    xtickfontsize=12, ytickfontsize=12, linewidth=2,
    xlim=(0,12), ylim=(0,1), label="4dt", legend=false,
    top_margin = 0mm, bottom_margin = 8mm, left_margin = 12mm, right_margin = 2mm)
plot!(r2[:,1], r2[:,3], linewidth=2, label="8dt", legendfontsize=12)
plot!(r2[:,1], r2[:,4], linewidth=2, label="12dt", legendfontsize=12)
plot!(r2[:,1], r2[:,5], linewidth=2, label="16dt", legendfontsize=12)
plot!(r2[:,1], r2[:,6], linewidth=2, label="32dt", legendfontsize=12)
plot!(size=(500,400))
savefig("r2_plot_season.pdf")


max_cfl_months = readdlm("max_cfl_months.txt") 

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plot(months, max_cfl_months, xlabel="Months", ylabel="Maximum CFL number", linewidth=3,
xtickfontsize=12, ytickfontsize=12, xlabelfontsize=14, ylabelfontsize=14, legend=false,
top_margin = 0mm, bottom_margin = 10mm, left_margin = 12mm, right_margin = 0mm,
ylim=(0, 1))
plot!(size=(500,400))
savefig("max_cfl_months.pdf")
