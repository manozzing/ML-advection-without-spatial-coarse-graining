#### This Julia file contains all necessary functions to run the machine learned advection operator in 1-D and 2-D domain.
#### This file also contains the training function.
#### There are two different Julia package you can use for this solver. Flux.jl is the default machine learning package.
#### We also use SimpleChains.jl because it is faster than Flux.jl. 
#### We trained the model parameters with Flux.jl and then transferred the model parameters to SimpleChains.jl model. 

## Step filter function
function step!(x)
    if x > 0
        x = 1
    else
        x = 0
    end
end

## Integrating a single timestep using ML based advection solver ##
## This code is called for model training later ##
## Integrate function with 3 stencil (default) ##
function one_step_flux(x, cfl, model, xdim, temp_factor)
    ## Initialize
    smax = maximum(x)
    cflmax = maximum(abs.(cfl))
    cflmean = mean(abs.(cfl))
    if smax > 0
        x = x / smax
    end

    if temp_factor == 4
        cfl = cfl / cflmax
        x_bc = zeros(Float32, xdim+6); cfl_bc = zeros(Float32, xdim+6)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4:xdim+3] = x; x_bc[xdim+4] = x[xdim]; x_bc[xdim+5] = x[xdim]; x_bc[xdim+6] = x[xdim];
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4:xdim+3] = cfl; cfl_bc[xdim+4] = cfl[xdim]; cfl_bc[xdim+5] = cfl[xdim]; cfl_bc[xdim+6] = cfl[xdim];
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+6, 1, 2, 1))), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 8
        cfl = cfl / cflmax
        x_bc = zeros(Float32, xdim+8); cfl_bc = zeros(Float32, xdim+8)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4] = x[1]; x_bc[5:xdim+4] = x; x_bc[xdim+5] = x[xdim]; x_bc[xdim+6] = x[xdim]; x_bc[xdim+7] = x[xdim]; x_bc[xdim+8] = x[xdim];
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4] = cfl[1]; cfl_bc[5:xdim+4] = cfl; cfl_bc[xdim+5] = cfl[xdim]; cfl_bc[xdim+6] = cfl[xdim]; cfl_bc[xdim+7] = cfl[xdim]; cfl_bc[xdim+8] = cfl[xdim];
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+8, 1, 2, 1))), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 12 || temp_factor == 16
        cfl = cfl / cflmax
        x_bc = zeros(Float32, xdim+12); cfl_bc = zeros(Float32, xdim+12)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4] = x[1]; x_bc[5] = x[1]; x_bc[6] = x[1];
        x_bc[7:xdim+6] = x; 
        x_bc[xdim+7] = x[xdim]; x_bc[xdim+8] = x[xdim]; x_bc[xdim+9] = x[xdim]; x_bc[xdim+10] = x[xdim]; x_bc[xdim+11] = x[xdim]; x_bc[xdim+12] = x[xdim]
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4] = cfl[1]; cfl_bc[5] = cfl[1]; cfl_bc[6] = cfl[1]; 
        cfl_bc[7:xdim+6] = cfl; 
        cfl_bc[xdim+7] = cfl[xdim]; cfl_bc[xdim+8] = cfl[xdim]; cfl_bc[xdim+9] = cfl[xdim]; cfl_bc[xdim+10] = cfl[xdim]; cfl_bc[xdim+11] = cfl[xdim]; cfl_bc[xdim+12] = cfl[xdim];
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+12, 1, 2, 1))), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 32
        cfl = cfl / cflmean
        x_bc = zeros(Float32, xdim+16); cfl_bc = zeros(Float32, xdim+16)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4] = x[1]; x_bc[5] = x[1]; x_bc[6] = x[1]; x_bc[7] = x[1]; x_bc[8] = x[1];
        x_bc[9:xdim+8] = x; 
        x_bc[xdim+9] = x[xdim]; x_bc[xdim+10] = x[xdim]; x_bc[xdim+11] = x[xdim]; x_bc[xdim+12] = x[xdim]; x_bc[xdim+13] = x[xdim]; x_bc[xdim+14] = x[xdim]; x_bc[xdim+15] = x[xdim]; x_bc[xdim+16] = x[xdim];
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4] = cfl[1]; cfl_bc[5] = cfl[1]; cfl_bc[6] = cfl[1]; cfl_bc[7] = cfl[1]; cfl_bc[8] = cfl[1]; 
        cfl_bc[9:xdim+8] = cfl; 
        cfl_bc[xdim+9] = cfl[xdim]; cfl_bc[xdim+10] = cfl[xdim]; cfl_bc[xdim+11] = cfl[xdim]; cfl_bc[xdim+12] = cfl[xdim]; cfl_bc[xdim+13] = cfl[xdim]; cfl_bc[xdim+14] = cfl[xdim]; cfl_bc[xdim+15] = cfl[xdim]; cfl_bc[xdim+16] = cfl[xdim];
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+16, 1, 2, 1))), (xdim, 2)) * smax * cflmean
    end
    return flux_estimated
end

## Integrating a single timestep using ML based advection solver ##
## This code is called for model training later ##
## Integrate function with 3 stencil (default) ##
function one_step_flux_train(x, cfl, model, xdim, temp_factor)
    ## Initialize
    smax = maximum(x)
    cflmean = mean(abs.(cfl))
    cflmax = maximum(abs.(cfl))
    if smax > 0
        x = x / smax
    end
    
    if temp_factor == 4
        cfl = cfl / cflmax
        x_bc = vcat( [x[1]], [x[1]], [x[1]], [x[i] for i in 1:xdim], [x[xdim]], [x[xdim]], [x[xdim]])
        cfl_bc = vcat( [cfl[1]], [cfl[1]], [cfl[1]], [cfl[i] for i in 1:xdim], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]])
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+6, 1, 2, 1))), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 8
        cfl = cfl / cflmax
        x_bc = vcat( [x[1]], [x[1]], [x[1]], [x[1]], [x[i] for i in 1:xdim], [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]])
        cfl_bc = vcat( [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[i] for i in 1:xdim], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]])
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+8, 1, 2, 1))), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 12 || temp_factor == 16
        cfl = cfl / cflmax
        x_bc = vcat( [x[1]], [x[1]], [x[1]], [x[1]], [x[1]], [x[1]], [x[i] for i in 1:xdim], 
                [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]])
        cfl_bc = vcat( [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[i] for i in 1:xdim], 
                [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]])
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+12, 1, 2, 1))), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 32
        cfl = cfl / cflmean
        x_bc = vcat( [x[1]], [x[1]], [x[1]], [x[1]], [x[1]], [x[1]], [x[1]], [x[1]], [x[i] for i in 1:xdim], 
                [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]], [x[xdim]])
        cfl_bc = vcat( [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[1]], [cfl[i] for i in 1:xdim], 
                [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]], [cfl[xdim]])
        flux_estimated = reshape(model(reshape(hcat(x_bc, cfl_bc), (xdim+16, 1, 2, 1))), (xdim, 2)) * smax * cflmean
    end

    return flux_estimated
end


#### Learned advection in 1-D
## This is to make splitted scheme
function advect1d_ml(s1d_in,cfl,nz,model,MP)
    s1_input = zeros(Float32,nz,1,1); s1d_out = zeros(Float32,nz); s1_input[:,1,1] .= s1d_in[:]
    filter = zeros(Float32, nz, 2); filter[:,1] = MP(s1_input)[:,1,1]
    filter[:,1] .= step!.(filter[:,1]); filter[:,2] .= filter[:,1]
    flux_estimated = one_step_flux(s1_input, cfl, model, nz) .* filter
    s1d_out .= flux_estimated[:,1] .- flux_estimated[:,2]
    return s1d_out
end

#### Learned advection in 2-D
## copy outputs of advect1d and then use them for non-directional 2-D advection
function advect2D_ml(dt::Float32, nx::Int, ny::Int, s1::Matrix{Float32}, u::Matrix{Float32}, v::Matrix{Float32}, model, MP)
    s2d = zeros(Float32,nx,ny); s2d[1:nx,1:ny] .= s1'
    u_bc = zeros(Float32, nx,ny); v_bc = zeros(Float32, nx,ny); u_bc[1:nx,1:ny] .= u'; v_bc[1:nx,1:ny] .= v'
    s1d_xin = zeros(Float32, nx); s1d_xout = zeros(Float32, nx)
    s1d_yin = zeros(Float32, ny); s1d_yout = zeros(Float32, ny)
    cfl_x1d = zeros(Float32, nx, ny); cfl_y1d = zeros(Float32, nx, ny)
    F_Q = zeros(Float32, nx, ny); G_Q = zeros(Float32, nx, ny); F_GQ = zeros(Float32, nx, ny); G_FQ = zeros(Float32, nx, ny)
    Re = 6378000; dx = zeros(Float32, ny); dy = Float32(Re*(lats[2] - lats[1])*π/180)
    
    dx[:] .= Float32.(Re * cos.(lats[:]*π/180) * (lons[2] - lons[1])*π/180)
    cfl_x1d .= u_bc * dt
    cfl_y1d .= v_bc * dt / dy
    for j in 1:ny
        F_Q[:,j] = advect1d_ml(s2d[:,j],cfl_x1d[:,j]/dx[j],nx,model,MP)
    end
    for i in 1:nx
        G_Q[i,:] = advect1d_ml(s2d[i,:],cfl_y1d[i,:],ny,model,MP)
    end    
    for j in 1:ny
        F_GQ[:,j] = advect1d_ml(s2d[:,j] .+ 0.5*G_Q[:,j],cfl_x1d[:,j]/dx[j],nx,model,MP)
    end    
    for i in 1:nx
        G_FQ[i,:] = advect1d_ml(s2d[i,:] .+ 0.5*F_Q[i,:],cfl_y1d[i,:],ny,model,MP)
    end    
    s1 .= s2d' .+ F_GQ' .+ G_FQ'
    return s1     
end

#### A function programming the learned advection scheme for 2-D implementation
function pgm_ml_2D(u, nstep::Int, u_wind, v_wind, model, MP, nx, ny)
    dt = Float32(300)*temp_factor
    ## Integrate ## You can change the velocity field per the resolution of your interest
    ilev=1; ispec=1;
    for n in 1:nstep-1
        u[n+1,:,:,ilev,ispec] = advect2D_ml(dt, nx, ny,
                u[n,:,:,ilev,ispec], u_wind[n,:, :, ilev], v_wind[n, :, :, ilev], model, MP)
        u[n+1,:,:,ilev,ispec] .= u[n+1,:,:,ilev,ispec] .* (u[n+1,:,:,ilev,ispec] .> 0)
    end
    return u
end

## Integrating a single timestep using ML based advection solver ##
## This code is called for model training later ##
## Integrate function with 3 stencil (default) ##
function one_step_flux_SC(x, cfl, model_SC, p_SC, xdim)
    ## Initialize
    smax = maximum(x)
    cflmax = maximum(abs.(cfl))
    cflmean = mean(abs.(cfl)) ## Only for 32dt
    if smax > 0
        x = x / smax
    end

    if temp_factor == 4
        cfl = cfl / cflmax
        x_bc = zeros(Float32, xdim+6); cfl_bc = zeros(Float32, xdim+6)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4:xdim+3] = x; x_bc[xdim+4] = x[xdim]; x_bc[xdim+5] = x[xdim]; x_bc[xdim+6] = x[xdim];
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4:xdim+3] = cfl; cfl_bc[xdim+4] = cfl[xdim]; cfl_bc[xdim+5] = cfl[xdim]; cfl_bc[xdim+6] = cfl[xdim];
        flux_estimated = reshape(model_SC(reshape(hcat(x_bc, cfl_bc), (xdim+6, 1, 2, 1)), p_SC), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 8
        cfl = cfl / cflmax
        x_bc = zeros(Float32, xdim+8); cfl_bc = zeros(Float32, xdim+8)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4] = x[1]; x_bc[5:xdim+4] = x; x_bc[xdim+5] = x[xdim]; x_bc[xdim+6] = x[xdim]; x_bc[xdim+7] = x[xdim]; x_bc[xdim+8] = x[xdim];
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4] = cfl[1]; cfl_bc[5:xdim+4] = cfl; cfl_bc[xdim+5] = cfl[xdim]; cfl_bc[xdim+6] = cfl[xdim]; cfl_bc[xdim+7] = cfl[xdim]; cfl_bc[xdim+8] = cfl[xdim];
        flux_estimated = reshape(model_SC(reshape(hcat(x_bc, cfl_bc), (xdim+8, 1, 2, 1)), p_SC), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 12 || temp_factor == 16
        cfl = cfl / cflmax
        x_bc = zeros(Float32, xdim+12); cfl_bc = zeros(Float32, xdim+12)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4] = x[1]; x_bc[5] = x[1]; x_bc[6] = x[1];
        x_bc[7:xdim+6] = x; 
        x_bc[xdim+7] = x[xdim]; x_bc[xdim+8] = x[xdim]; x_bc[xdim+9] = x[xdim]; x_bc[xdim+10] = x[xdim]; x_bc[xdim+11] = x[xdim]; x_bc[xdim+12] = x[xdim]; 
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4] = cfl[1]; cfl_bc[5] = cfl[1]; cfl_bc[6] = cfl[1];
        cfl_bc[7:xdim+6] = cfl; 
        cfl_bc[xdim+7] = cfl[xdim]; cfl_bc[xdim+8] = cfl[xdim]; cfl_bc[xdim+9] = cfl[xdim]; cfl_bc[xdim+10] = cfl[xdim]; cfl_bc[xdim+11] = cfl[xdim]; cfl_bc[xdim+12] = cfl[xdim];
        flux_estimated = reshape(model_SC(reshape(hcat(x_bc, cfl_bc), (xdim+12, 1, 2, 1)), p_SC), (xdim, 2)) * smax * cflmax
    elseif temp_factor == 32
        cfl = cfl / cflmean
        x_bc = zeros(Float32, xdim+16); cfl_bc = zeros(Float32, xdim+16)
        x_bc[1] = x[1]; x_bc[2] = x[1]; x_bc[3] = x[1]; x_bc[4] = x[1]; x_bc[5] = x[1]; x_bc[6] = x[1]; x_bc[7] = x[1]; x_bc[8] = x[1]; 
        x_bc[9:xdim+8] = x; 
        x_bc[xdim+9] = x[xdim]; x_bc[xdim+10] = x[xdim]; x_bc[xdim+11] = x[xdim]; x_bc[xdim+12] = x[xdim]; x_bc[xdim+13] = x[xdim]; x_bc[xdim+14] = x[xdim]; x_bc[xdim+15] = x[xdim]; x_bc[xdim+16] = x[xdim];
        cfl_bc[1] = cfl[1]; cfl_bc[2] = cfl[1]; cfl_bc[3] = cfl[1]; cfl_bc[4] = cfl[1]; cfl_bc[5] = cfl[1]; cfl_bc[6] = cfl[1]; cfl_bc[7] = cfl[1]; cfl_bc[8] = cfl[1]; 
        cfl_bc[9:xdim+8] = cfl; 
        cfl_bc[xdim+9] = cfl[xdim]; cfl_bc[xdim+10] = cfl[xdim]; cfl_bc[xdim+11] = cfl[xdim]; cfl_bc[xdim+12] = cfl[xdim]; cfl_bc[xdim+13] = cfl[xdim]; cfl_bc[xdim+14] = cfl[xdim]; cfl_bc[xdim+15] = cfl[xdim]; cfl_bc[xdim+16] = cfl[xdim];
        flux_estimated = reshape(model_SC(reshape(hcat(x_bc, cfl_bc), (xdim+16, 1, 2, 1)), p_SC), (xdim, 2)) * smax * cflmax
    end
    return flux_estimated
end


#### Learned advection in 1-D
## This is to make splitted scheme
function advect1d_ml_SC(s1d_in,cfl,nz,model_SC,p_SC,MP)
    s1_input = zeros(Float32,nz,1,1); s1d_out = zeros(Float32,nz); s1_input[:,1,1] .= s1d_in[:]
    filter = zeros(Float32, nz, 2); filter[:,1] = MP(s1_input)[:,1,1]
    filter[:,1] .= step!.(filter[:,1]); filter[:,2] .= filter[:,1]
    flux_estimated = one_step_flux_SC(s1_input, cfl, model_SC, p_SC, nz) .* filter
    s1d_out .= flux_estimated[:,1] .- flux_estimated[:,2]
    return s1d_out
end

#### Learned advection in 2-D
## copy outputs of advect1d and then use them for non-directional 2-D advection
function advect2D_ml_SC(dt::Float32, nx::Int, ny::Int, s1::Matrix{Float32}, u::Matrix{Float32}, v::Matrix{Float32}, model_SC, p_SC, MP)
    s2d = zeros(Float32,nx,ny); s2d[1:nx,1:ny] .= s1'
    u_bc = zeros(Float32, nx,ny); v_bc = zeros(Float32, nx,ny); u_bc[1:nx,1:ny] .= u'; v_bc[1:nx,1:ny] .= v'
    s1d_xin = zeros(Float32, nx); s1d_xout = zeros(Float32, nx)
    s1d_yin = zeros(Float32, ny); s1d_yout = zeros(Float32, ny)
    cfl_x1d = zeros(Float32, nx, ny); cfl_y1d = zeros(Float32, nx, ny)
    F_Q = zeros(Float32, nx, ny); G_Q = zeros(Float32, nx, ny); F_GQ = zeros(Float32, nx, ny); G_FQ = zeros(Float32, nx, ny)
    Re = 6378000; dx = zeros(Float32, ny); dy = Float32(Re*(lats[2] - lats[1])*π/180)
    
    dx[:] .= Float32.(Re * cos.(lats[:]*π/180) * (lons[2] - lons[1])*π/180)
    cfl_x1d .= u_bc * dt
    cfl_y1d .= v_bc * dt / dy
    for j in 1:ny
        F_Q[:,j] = advect1d_ml_SC(s2d[:,j],cfl_x1d[:,j]/dx[j],nx,model_SC,p_SC,MP)
    end
    for i in 1:nx
        G_Q[i,:] = advect1d_ml_SC(s2d[i,:],cfl_y1d[i,:],ny,model_SC,p_SC,MP)
    end    
    for j in 1:ny
        F_GQ[:,j] = advect1d_ml_SC(s2d[:,j] .+ 0.5*G_Q[:,j],cfl_x1d[:,j]/dx[j],nx,model_SC,p_SC,MP)
    end    
    for i in 1:nx
        G_FQ[i,:] = advect1d_ml_SC(s2d[i,:] .+ 0.5*F_Q[i,:],cfl_y1d[i,:],ny,model_SC,p_SC,MP)
    end    
    s1 .= s2d' .+ F_GQ' .+ G_FQ'
    return s1     
end

#### A function programming the learned advection scheme for 2-D implementation
function pgm_ml_SC_2D(u, nstep::Int, u_wind, v_wind, model_SC, p_SC, MP, nx, ny)
    dt = Float32(300)*temp_factor
    ## Integrate ## You can change the velocity field per the resolution of your interest
    ilev=1; ispec=1;
    for n in 1:nstep-1
        u[n+1,:,:,ilev,ispec] = advect2D_ml_SC(dt, nx, ny,
                u[n,:,:,ilev,ispec], u_wind[n,:, :, ilev], v_wind[n, :, :, ilev], model_SC, p_SC, MP)
        u[n+1,:,:,ilev,ispec] .= u[n+1,:,:,ilev,ispec] .* (u[n+1,:,:,ilev,ispec] .> 0)
    end
    return u
end


function advect_ml!(u, start, finish, h)
    u_wind=zeros(Float32, length(lats),length(lons),length(levs))
    v_wind=zeros(Float32, length(lats),length(lons),length(levs))
    Δt = Float32(finish - start)

    for i in 1:length(lons)
        for j in 1:length(lats)
            for k in 1:length(levs)
                u_wind[j,i,k] = Float32.(EarthSciData.interp!(u_itp, start, lons[i], lats[j], levs[k]));
                v_wind[j,i,k] = Float32.(EarthSciData.interp!(v_itp, start, lons[i], lats[j], levs[k]));
            end
        end
    end

    for ilev in eachindex(levs)       
        for ispec in 1:1
            u[h,:,:,ilev,ispec] = advect2D_ml(Δt, length(lons), length(lats), 
                u[h,:,:,ilev,ispec], u_wind[:, :, ilev], v_wind[:, :, ilev], model, MP)
        end
    end

end




## Training function for 5 time steps training
function five_step_train!(ps, data, opt, dz, dt)
    local len = length(data)
    #len_5 = len÷5; rand_num = rand(1:len-len_5+1); data_batch = data[rand_num:rand_num+len_5-1]; 
    xdim = size(data[1][1],1);
    for i in 1:len
        ii = rand(1:len)
        if sum(data[ii][1][:,1,:]) > 1e-5
            gs = gradient(ps) do
                if  0 < ii < len-3
                    A = hcat([one_step_flux_train(data[ii][1][:,1,:], data[ii][1][:,2,:], model, xdim)])
                    x = data[ii][1][:,1]
                    for j in 1:4
                        x = x + A[j][1:xdim] - A[j][2:xdim+1]
                        y = data[ii+j][1][:,2]
                        A = hcat(A, [one_step_flux_train(hcat(x, y)[:,1,:], hcat(x, y)[:,2,:], model, xdim)])
                    end
                    B = hcat([data[ii+k-1][2] for k in 1:5])
                    training_loss = mean(Flux.Losses.mae.(A, B))
                elseif len-3 ≤ ii < len
                    n = len - ii
                    A = hcat([one_step_flux_train(data[ii][1][:,1,:], data[ii][1][:,2,:], model, xdim)])
                    x = data[ii][1][:,1]
                    for j in 1:n
                        x = x + A[j][1:xdim] - A[j][2:xdim+1]
                        y = data[ii+j][1][:,2]
                        A = hcat(A, [one_step_flux_train(hcat(x, y)[:,1,:], hcat(x, y)[:,2,:], model, xdim)])
                    end
                    B = hcat([data[ii+k][2] for k in 0:n])
                    training_loss = mean(Flux.Losses.mae.(A, B))
                elseif ii == len
                    A = hcat([one_step_flux_train(data[ii][1][:,1,:], data[ii][1][:,2,:], model, xdim)])
                    B = data[ii][2]
                    training_loss = mean(Flux.Losses.mae.(A, B))
                end
                return training_loss
            end
        end
        Flux.update!(opt, ps, gs)
    end
end



function one_to_three_index(one_dim_index, size1=length(lats), size2=length(lons), size3=length(levs))
    i = (one_dim_index-1) % size3 + 1; j = ((one_dim_index-1) ÷ size3) % size2 + 1; k = (one_dim_index-1) ÷ (size3 * size2) + 1
    return (k, j, i)
end
function prob_func(prob, i, repeat)
    p = [defaults[p] for p in parameters(sys)]; ii = one_to_three_index(i)
    remake(prob, u0 = u[h, ii[1], ii[2], ii[3], :], p = p, tspan = (start, finish))
end
function reduction_func(u, data, I)
    for (iii, i) in enumerate(I)
        ii = one_to_three_index(i); u[h+1, ii[1], ii[2], ii[3], :] .= data[iii]
    end
    return u, false
end