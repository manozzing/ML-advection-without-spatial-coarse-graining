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
temp_factor = 16

## You will confirm the coarse-graining factor in both space and time.
println("space factor: "*string(space_factor)*"   temp factor: "*string(temp_factor))

## Set the simulation domain
ModelingToolkit.check_units(eqs...) = nothing; start = Dates.datetime2unix(Dates.DateTime(2018, 1, 1))
@parameters t
lons = collect(-130.0:0.3125*space_factor:-60.0); lats = collect(9.75:0.25*space_factor:60.0); levs = collect(1:1:1)
nsims = length(lats)*length(lons)*length(levs)
ntimesteps = numsteps÷temp_factor+1; s_per_timestep = 300*temp_factor; finish = start + s_per_timestep
model_ode = SuperFast(t) + FastJX(t); sys = structural_simplify(get_mtk(model_ode));
prob = ODEProblem(sys, [], (start, finish), []); sol = solve(prob, TRBDF2(), saveat=1800.0)
geos = GEOSFP{Float64}("0.25x0.3125_NA", t; coord_defaults = Dict(:lat => 34.0, :lon=>-100.0, :lev => 1));
latgrid = reshape(repeat(lats, outer = length(lons) * length(levs)), length(lats), length(lons), length(levs))
longrid = reshape(repeat(lons, outer = length(levs), inner = length(lats)), length(lats), length(lons), length(levs))
levgrid = reshape(repeat(levs, inner = length(lats) * length(lons)), length(lats), length(lons), length(levs))
defaults = ModelingToolkit.get_defaults(sys); vars = states(sys); ps = parameters(sys); indexof(sym, syms) = findfirst(isequal(sym), syms);

# Load the velocity data in the given spatial domain
Re = 6378000; vert_pressure_hpa = [1013.250]; Δy = Re*(lats[2] - lats[1])*π/180; fs = geos.filesets["A3dyn"]; default_time = Dates.DateTime(2018, 1, 1)
u_itp = EarthSciData.DataSetInterpolator{Float64}(fs, "U", default_time);
v_itp = EarthSciData.DataSetInterpolator{Float64}(fs, "V", default_time); 
start = Dates.datetime2unix(Dates.DateTime(2018, 1, 1)); h = 1
u_wind=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs)); v_wind=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs))

times = []
for h in 1:ntimesteps
    push!(times, start);
    for i in 1:length(lons)
        for j in 1:length(lats)
            u_wind[h,j,i,1] = Float32.(EarthSciData.interp!(u_itp, start, lons[i], lats[j], 1.0));
            v_wind[h,j,i,1] = Float32.(EarthSciData.interp!(v_itp, start, lons[i], lats[j], 1.0));
        end
    end
    start += s_per_timestep
    finish += s_per_timestep
end
push!(times, finish);

## Load the 2-D testing dataset.
ds_2D = NCDataset("/projects/illinois/eng/cee/ctessum/manhop2/Species_reduction/2D_test_nested_1lev_"*string(days)*"day_Jan2018_"*string(space_factor)*"x"*string(temp_factor)*"t_+1.nc")
u_reference = zeros(Float32, 10*24*12÷temp_factor+1, length(lats), length(lons), 1, 1); u_reference .= ds_2D["u"]

## Build a model
model, MP = Flux_model_params(space_factor, temp_factor)
loss(x, y) = Flux.Losses.mae(model(x), y); ps = Flux.params(model);

## Concentration array to test the 2-D advection.
u_learned = zeros(Float32, 10*24*12÷temp_factor+1, length(lats), length(lons), length(levs), 1)
for j in length(lats)÷3:length(lats)÷3*2
    for i in length(lons)÷3:length(lons)÷3*2
        u_learned[1,i,j,1,1] = 100
    end
end

## Load the trained model parameters
@load "../data/model_params/CNN_trained_"*string(temp_factor)*"dt.bson_MODEL" model
ps = Flux.params(model);
model_SC, p_SC = param_Flux_to_SC(temp_factor, ps)

## Model evaluation in 2-D advection.
#@time u_learned = pgm_ml_2D(u_learned, 10*24*12÷temp_factor, u_wind, v_wind, model, MP, length(lons), length(lats));
@time u_learned = pgm_ml_SC_2D(u_learned, 10*24*12÷temp_factor+1, u_wind, v_wind, model_SC, p_SC, MP, length(lons), length(lats));
mae_2D_total = StatsBase.L1dist(reshape(u_learned[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))/(10*24*12÷temp_factor*length(lats)*length(lons))
rmse_2D_total = StatsBase.L2dist(reshape(u_learned[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))/sqrt(10*24*12÷temp_factor*length(lats)*length(lons))
r²_2D_total = Statistics.cor(reshape(u_learned[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))^2
println(mae_2D_total, "    ", rmse_2D_total, "    ", r²_2D_total)

## Benchmarking the timing
@benchmark pgm_ml_SC_2D(u_learned, 10*24*12÷temp_factor, u_wind, v_wind, model_SC, p_SC, MP, length(lons), length(lats))

## Error statistics
mae_2D = zeros(Float32, 10*24*12÷temp_factor+1)
rmse_2D = zeros(Float32, 10*24*12÷temp_factor+1)
r_2D = zeros(Float32, 10*24*12÷temp_factor+1)
r²_2D = zeros(Float32, 10*24*12÷temp_factor+1)
mass_2D = zeros(Float32, 10*24*12÷temp_factor+1)
sum_original = sum(u_learned[1,:,:,:,:])

for timet in 1:10*24*12÷temp_factor+1
    mae_2D[timet] = StatsBase.L1dist(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))/(length(lats)*length(lons))
    rmse_2D[timet] = StatsBase.L2dist(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))/sqrt(length(lats)*length(lons))
    r_2D[timet] = Statistics.cor(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))
    r²_2D[timet] = Statistics.cor(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))^2
    mass_2D[timet] = sum(u_learned[timet,:,:,:,:]) / sum_original
    println("time: ", timet, "   MAE = ", mae_2D[timet], "   RMSE = ", rmse_2D[timet], "   R = ", r_2D[timet], "   R² = ", r²_2D[timet])
    println("time: ", timet, "   mass = ", mass_2D[timet])
end

## Save error statistics
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/mae_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", mae_2D)
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/rmse_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", rmse_2D)
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/r_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", r_2D)
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/r2_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", r²_2D)
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/mass_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", mass_2D)

## Save the advection output to netcdf format
ds = NCDataset("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/2D_learned_output_1level_"*string(days)*"day_Jan2018_"*string(space_factor)*"x"*string(temp_factor)*"t.nc","c")

defDim(ds,"time",numsteps÷temp_factor+1)#12*24*30)
defDim(ds,"lat",length(lats))
defDim(ds,"lon",length(lons))#÷space_factor+1)
defDim(ds,"lev",1)
defDim(ds,"spec",1)

conc = defVar(ds,"u",Float32,("time","lat","lon","lev","spec"))
for t in 1:numsteps÷temp_factor+1#
    conc[t,:,:,:,:] = u_learned[t,:,:,:,:]#u_temp[t,:,:,:,:]
end
close(ds)


#### Visualization for paper
makemap_level(u_learned,1,1,2000)
savefig("figs_paper/"*string(space_factor)*"x"*string(temp_factor)*"dt_day0.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*1,1,2000)
savefig("figs_paper/"*string(space_factor)*"x"*string(temp_factor)*"dt_day2.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*2,1,2000)
savefig("figs_paper/"*string(space_factor)*"x"*string(temp_factor)*"dt_day4.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*3,1,2000)
savefig("figs_paper/"*string(space_factor)*"x"*string(temp_factor)*"dt_day6.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*4,1,2000)
savefig("figs_paper/"*string(space_factor)*"x"*string(temp_factor)*"dt_day8.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*5,1,2000)
savefig("figs_paper/"*string(space_factor)*"x"*string(temp_factor)*"dt_day10.png")

makemap_level(u_reference,1,1,2000)
savefig("figs_paper/reference_day0.png")

makemap_level(u_reference,1+10*24*12÷temp_factor÷5*1,1,2000)
savefig("figs_paper/reference_day2.png")

makemap_level(u_reference,1+10*24*12÷temp_factor÷5*2,1,2000)
savefig("figs_paper/reference_day4.png")

makemap_level(u_reference,1+10*24*12÷temp_factor÷5*3,1,2000)
savefig("figs_paper/reference_day6.png")

makemap_level(u_reference,1+10*24*12÷temp_factor÷5*4,1,2000)
savefig("figs_paper/reference_day8.png")

makemap_level(u_reference,1+10*24*12÷temp_factor÷5*5,1,2000)
savefig("figs_paper/reference_day10.png")

