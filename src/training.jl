import Pkg
Pkg.activate("../.")

add EarthSciData@0.4.4
add EarthSciMLBase@0.8.0
add GasChem@0.5.1
add DomainSets, ModelingToolkit, DifferentialEquations, Dates
add Plots, Statistics, GeoDataFrames, NCDatasets, StatsBase
add Flux, BenchmarkTools
add BSON
add LoopVectorization, Tullio, Random

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

include("advection_operators.jl")
include("Flux_to_SC_params.jl")
include("visualization_utils.jl")

## Set the time domain and the spatiotemporal resolution
days = 10
numsteps = days*24*12
space_factor = 1
temp_factor = 8 ## Possible options are 4, 8, 12, 16, 32

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

## Convert the velocity to the CFL number. The CFL numbers will be used for the solver training.
cfl_x=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs)); cfl_y=zeros(Float32,ntimesteps,length(lats),length(lons),length(levs));
dy = Re*(lats[2] - lats[1])*π/180
for i in eachindex(lats)
    dx = Re * cos(lats[i]*π/180) * (lons[2] - lons[1])*π/180
    cfl_x[:,i,:,:] = u_wind[:,i,:,:] * s_per_timestep /dx
end
for j in eachindex(lons)
    cfl_y[:,:,j,:] = v_wind[:,:,j,:] * s_per_timestep /dy
end

## Load the training and testing dataset. Training dataset is 1-D advection results and testing dataset is 2-D advection results. 
ds_EW = NCDataset("../Species_reduction/1D_E-W_nested_1lev_"*string(days)*"day_Jan2018_"*string(space_factor)*"x"*string(temp_factor)*"t.nc");
ds_NS = NCDataset("../Species_reduction/1D_N-S_nested_1lev_"*string(days)*"day_Jan2018_"*string(space_factor)*"x"*string(temp_factor)*"t.nc");
ds_2D = NCDataset("../Species_reduction/2D_test_nested_1lev_"*string(days)*"day_Jan2018_"*string(space_factor)*"x"*string(temp_factor)*"t_+1.nc")

## Load the netcdf training/testing data into Julia arrays.
u_reference = zeros(Float32, 10*24*12÷temp_factor+1, length(lats), length(lons), 1, 1); u_reference .= ds_2D["u"]
Data_EW = []; nstep = 2; lat = 1
cfl1D = cfl_x[1:ntimesteps-1,lat,:,1]; scalar1D = ds_EW["u"][1:ntimesteps-1,lat,:,1,1]; flux1D = ds_EW["fe"][2:ntimesteps,lat,:,1,1]
nstep = size(cfl1D, 1); xdim = size(cfl1D, 2); history = (scalar1D, cfl1D);
scalar_NN_integrate = zeros(Float32, xdim, 1, 1, nstep); cfl_NN_integrate = zeros(Float32, xdim, 1, 1, nstep); flux_NN_integrate = zeros(Float32, xdim, 2, 1, nstep);
for i in 1:cld(202,space_factor)
    push!(Data_EW, (hcat(scalar_NN_integrate[:,:,:,1], cfl_NN_integrate[:,:,:,1]), flux_NN_integrate[:,:,:,1]))
end
Data_NS = []; nstep = 2; lon = 1
cfl1D = cfl_y[1:ntimesteps-1,lat,:,1]; scalar1D = ds_NS["u"][1:ntimesteps-1,lat,:,1,1]; flux1D = ds_NS["fe"][2:ntimesteps,lat,:,1,1] #/ -1
nstep = size(cfl1D, 1); ydim = size(cfl1D, 2); history = (scalar1D, cfl1D);
scalar_NN_integrate = zeros(Float32, ydim, 1, 1, nstep); cfl_NN_integrate = zeros(Float32, ydim, 1, 1, nstep); flux_NN_integrate = zeros(Float32, ydim, 2, 1, nstep);
for i in 1:cld(225,space_factor)
    push!(Data_NS, (hcat(scalar_NN_integrate[:,:,:,1], cfl_NN_integrate[:,:,:,1]), flux_NN_integrate[:,:,:,1]))
end

## Feed the data into CNN-readable format.
Random.seed!(1)
### Data feeding processes -- each latitude
for lat in eachindex(lats)
    cfl1Dx = cfl_x[1:ntimesteps-1,lat,:,1]; scalar1Dx = ds_EW["u"][1:ntimesteps-1,lat,:,1,1]; flux1Dx = ds_EW["fe"][2:ntimesteps,lat,:,1,1]
    nstep = size(cfl1Dx, 1); xdim = size(cfl1Dx, 2); historyx = (scalar1Dx, cfl1Dx);
    scalar_NN_integratex = zeros(Float32, xdim, 1, 1, nstep); cfl_NN_integratex = zeros(Float32, xdim, 1, 1, nstep); flux_NN_integratex = zeros(Float32, xdim, 2, 1, nstep);
    scalar_NN_integratex[:,1,1,:] = historyx[1]'[:,1:nstep] + 3e-4*rand(xdim, nstep); cfl_NN_integratex[:,1,1,:] = historyx[2]'[:,1:nstep]
    flux_NN_integratex[:,1,1,:] = flux1Dx[:,1:xdim]'; flux_NN_integratex[:,2,1,:] = flux1Dx[:,2:xdim+1]'
    Data_EW[lat] = []
    for i in 1:nstep-1
        push!(Data_EW[lat], (hcat(scalar_NN_integratex[:,:,:,i], cfl_NN_integratex[:,:,:,i]), flux_NN_integratex[:,:,:,i]))
    end
end
### -- each longitude
for lon in eachindex(lons)
    cfl1Dy = cfl_y[1:ntimesteps-1,:,lon,1]; scalar1Dy = ds_NS["u"][1:ntimesteps-1,:,lon,1,1]; flux1Dy = ds_NS["fe"][2:ntimesteps,:,lon,1,1]
    nstep = size(cfl1Dy, 1); ydim = size(cfl1Dy, 2); historyy = (scalar1Dy, cfl1Dy);
    scalar_NN_integratey = zeros(Float32, ydim, 1, 1, nstep); cfl_NN_integratey = zeros(Float32, ydim, 1, 1, nstep); flux_NN_integratey = zeros(Float32, ydim, 2, 1, nstep);
    scalar_NN_integratey[:,1,1,:] = historyy[1]'[:,1:nstep] + 3e-4*rand(ydim, nstep); cfl_NN_integratey[:,1,1,:] = historyy[2]'[:,1:nstep]; 
    flux_NN_integratey[:,1,1,:] = flux1Dy[:,1:ydim]'; flux_NN_integratey[:,2,1,:] = flux1Dy[:,2:ydim+1]'
    Data_NS[lon] = []
    for i in 1:nstep-1
        push!(Data_NS[lon], (hcat(scalar_NN_integratey[:,:,:,i], cfl_NN_integratey[:,:,:,i]), flux_NN_integratey[:,:,:,i]))
    end
end

## Build the CNN model.
model, MP = Flux_model_params(space_factor, temp_factor)
loss(x, y) = Flux.Losses.mae(model(x), y); ps = Flux.params(model);

## Concentration array to test the 2-D advection.
u_learned = zeros(Float32, 10*24*12÷temp_factor+1, length(lats), length(lons), length(levs), 1)
for j in length(lats)÷3:length(lats)÷3*2
    for i in length(lons)÷3:length(lons)÷3*2
        u_learned[1,i,j,1,1] = 100
    end
end

### Training loop with 2-D testing in each epoch
Re = 6378000; dy = Re*(lats[2] - lats[1])*π/180; dt = Float32(300)*temp_factor
@time for i in 101:500
    α = (1/(1+1*i))*0.01
    for j in 1:1#cld(20, space_factor)
        lat = rand(1:cld(202, space_factor))
        dx = Re * cos(lats[lat]*π/180) * (lons[2] - lons[1])*π/180
        five_step_train!(ps, Data_EW[lat], Adam(α, (0.9, 0.999)), dx, dt)
        lon = rand(1:cld(225, space_factor))
        five_step_train!(ps, Data_NS[lon], Adam(α, (0.9, 0.999)), dy, dt)
    end
    u_learned = pgm_ml_2D(u_learned, 10*24*12÷temp_factor, u_wind, v_wind, model, MP, length(lons), length(lats))
    mae_2D_total = StatsBase.L1dist(reshape(u_learned, 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))/(10*24*12÷temp_factor*length(lats)*length(lons))
    rmse_2D_total = StatsBase.L2dist(reshape(u_learned, 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))/sqrt(10*24*12÷temp_factor*length(lats)*length(lons))
    r²_2D_total = Statistics.cor(reshape(u_learned, 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))^2
    println(i, "    ", mae_2D_total, "    ", rmse_2D_total, "    ", r²_2D_total)
    @save "CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/flux_automatedCNN"*string(i)*"EPOCHS_"*string(space_factor)*"x"*string(temp_factor)*"t_bc.bson_MODEL" model
end

## In case you want to load the trained model parameters and evaluate the performance.
Re = 6378000; dy = Re*(lats[2] - lats[1])*π/180; dt = Float32(300)*temp_factor
@time for i in 59:59
    @load "CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/flux_automatedCNN"*string(i)*"EPOCHS_"*string(space_factor)*"x"*string(temp_factor)*"t_bc.bson_MODEL" model
    u_learned = pgm_ml_2D(u_learned, 10*24*12÷temp_factor, u_wind, v_wind, model, MP, length(lons), length(lats))
    mae_2D_total = StatsBase.L1dist(reshape(u_learned, 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))/(10*24*12÷temp_factor*length(lats)*length(lons))
    rmse_2D_total = StatsBase.L2dist(reshape(u_learned, 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))/sqrt(10*24*12÷temp_factor*length(lats)*length(lons))
    r²_2D_total = Statistics.cor(reshape(u_learned, 10*24*12÷temp_factor*length(lats)*length(lons)), reshape(u_reference[1:10*24*12÷temp_factor,:,:,:,:], 10*24*12÷temp_factor*length(lats)*length(lons)))^2
    println(i, "    ", mae_2D_total, "    ", rmse_2D_total, "    ", r²_2D_total)
end

## When you load the previous parameters for training again, you would need to use code below.
@load "CNN_Pooling_Flux_output/1x4t_automated/cfl_normed_bc/flux_automatedCNN59EPOCHS_1x4t_bc.bson_MODEL" model
ps = Flux.params(model);

## Record the error statistics in each time step
mae_2D = zeros(Float32, 10*24*12÷temp_factor+1)
rmse_2D = zeros(Float32, 10*24*12÷temp_factor+1)
r_2D = zeros(Float32, 10*24*12÷temp_factor+1)
r²_2D = zeros(Float32, 10*24*12÷temp_factor+1)

for timet in 1:10*24*12÷temp_factor+1
    mae_2D[timet] = StatsBase.L1dist(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))/(length(lats)*length(lons))
    rmse_2D[timet] = StatsBase.L2dist(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))/sqrt(length(lats)*length(lons))
    r_2D[timet] = Statistics.cor(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))
    r²_2D[timet] = Statistics.cor(reshape(u_learned[timet,:,:,:,:], length(lats)*length(lons)), reshape(u_reference[timet,:,:,:,:], length(lats)*length(lons)))^2
    println("time: ", timet, "   MAE = ", mae_2D[timet], "   RMSE = ", rmse_2D[timet], "   R = ", r_2D[timet], "   R² = ", r²_2D[timet])
end

writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/mae_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", mae_2D)
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/rmse_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", rmse_2D)
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/r_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", r_2D)
writedlm("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/r2_2D_"*string(space_factor)*"x"*string(temp_factor)*"t.txt", r²_2D)


## Save the learned 2-D advection output
for level in 1:1
    ds = NCDataset("CNN_Pooling_Flux_output/"*string(space_factor)*"x"*string(temp_factor)*"t_automated/cfl_normed_bc/2D_learned_output_"*string(level)*"level_"*string(days)*"day_Jan2018_"*string(space_factor)*"x"*string(temp_factor)*"t.nc","c")

    defDim(ds,"time",numsteps÷temp_factor+1)#12*24*30)
    defDim(ds,"lat",length(lats))
    defDim(ds,"lon",length(lons))#÷space_factor+1)
    defDim(ds,"lev",1)
    defDim(ds,"spec",1)

    conc = defVar(ds,"u",Float32,("time","lat","lon","lev","spec"))

    for t in 1:numsteps÷temp_factor#
        conc[t,:,:,1,:] = u_learned[t,:,:,level,:]#u_temp[t,:,:,:,:]
    end
    close(ds)
end

#### Visualization part for paper
makemap_level(u_learned,1,1,2000)
savefig("figs_paper/1x4dt_day0.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*1,1,2000)
savefig("figs_paper/1x4dt_day2.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*2,1,2000)
savefig("figs_paper/1x4dt_day4.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*3,1,2000)
savefig("figs_paper/1x4dt_day6.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*4,1,2000)
savefig("figs_paper/1x4dt_day8.png")

makemap_level(u_learned,1+10*24*12÷temp_factor÷5*5,1,2000)
savefig("figs_paper/1x4dt_day10.png")
