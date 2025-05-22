coastlines = GeoDataFrames.read("../data/ne_110m_coastline/ne_110m_coastline.shp");

function makemap_level(u,i,j,k)
    heatmap(lons, lats, u[i, :, :, j, 1], 
        ylabel="Latitude (°)", xlabel="Longitude (°)", clim=(0,k),
        #colorbar_title="Mixing ratio (ppb)", 
        c=cgrad([:white,:orange,:red,:purple,:black]),
        tickfontsize = 9, colorbar_titlefontsize = 10,
        top_margin = 0mm, bottom_margin = 5mm, left_margin = 10mm, right_margin = 5mm)  #white,:yellow,:red
    plot!(coastlines.geometry, linecolor=:black, xlim=(minimum(lons), maximum(lons)), 
        ylim=(minimum(lats), maximum(lats)), 
        aspect_ratio=:equal)
        annotate!(-112, 12, Dates.unix2datetime(times[i]))
        #annotate!(-120, 16, "r² = "*string(round(r²_2D[i]; digits=2)))
end

function makemap(u,i,k)
    heatmap(lons, lats, u[i, :, :, 1, 1], 
        ylabel="Latitude (°)", xlabel="Longitude (°)", clim=(0,k),
        colorbar_title="Mixing ratio (ppb)", c=cgrad([:white,:orange,:red,:purple,:black]))  #white,:yellow,:red
    plot!(coastlines.geometry, linecolor=:black, xlim=(minimum(lons), maximum(lons)), 
        ylim=(minimum(lats), maximum(lats)), 
        aspect_ratio=:equal)
    annotate!(-112, 12, Dates.unix2datetime(times[i]))
    #annotate!(-112, 13, r²[i])
end

function makemap_flux(u,i)
    heatmap(lons, lats, FLUX[i, 1:length(lats), 1:length(lons), 1, 1], 
        ylabel="Latitude (°)", xlabel="Longitude (°)", clim=(0,10),
        colorbar_title="Mixing ratio (ppb)", c=cgrad([:white,:orange,:red,:purple,:black]))  #white,:yellow,:red
    plot!(coastlines.geometry, linecolor=:black, xlim=(minimum(lons), maximum(lons)), 
        ylim=(minimum(lats), maximum(lats)), 
        aspect_ratio=:equal)
    annotate!(-112, 12, Dates.unix2datetime(times[i]))
end
