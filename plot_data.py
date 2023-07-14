
import matplotlib.pyplot as plt
import numpy as np
import cartopy as cp
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from load_dataset import latitude, longitude, ice_thickness


# plot_data.py - UIT - Martin Stave - mvrtinstave@gmail.com
# this file plots the data arrays from the load_dataset.py file


def color_plot(): # pseudocolor plot in 2D

        # creating x and y arrays for plotting 
        lon_grid = np.arange(longitude.shape[0])
        lat_grid = np.arange(latitude.shape[0]) 
    
        plt.figure(figsize=(10,6)) # set figure size
        plt.pcolormesh(lon_grid, lat_grid, ice_thickness, cmap='jet') # creates grid of colored cells

        # add colorbar, title and axis labels
        plt.colorbar(label='Ice thickness (m)')
        plt.title('Arctic Region Sea Ice')
        plt.xlabel('Longitude (degrees east)')
        plt.ylabel('Latitude (degrees north)')

        # invert y axis to display map correctly and add grid
        plt.gca().invert_yaxis()
        plt.grid(True) 
        
        plt.show() 




def scatter_plot(): 
    # geographic plot centred around the North Pole
    # projection is used to convert points on a curved surface i.e. the earth
    # using scattering to display sea ice thickness
    
    # set figuresize and projection
    fig, ax = plt.subplots(1, 1, figsize=(8,8), 
                           subplot_kw = {'projection':cp.crs.NorthPolarStereo()})

    # add ocean and land outlines and colors
    ax.add_feature(cp.feature.OCEAN)
    ax.add_feature(cp.feature.LAND, edgecolor='black', zorder=1)
    ax.set_facecolor((1.0,1.0,1.0))
    ax.set_extent([-180,180,90,66], cp.crs.PlateCarree()) # set coordinate extent
    # changing extent will increase or decrease coverage

    # add gridlines displaying latitude and longitude angles
    ax.gridlines(draw_labels=True, alpha=0.5, color='gray', 
                 linestyle='-', linewidth=0.5, 
                 xlocs = np.arange(-180, 181, 30),
                 ylocs = np.arange(60, 91, 5))

    # visualize sea ice thickness as scatterpoints with corresponding longitude and latitude
    sc = ax.scatter(longitude, latitude, c=ice_thickness,
                    cmap='jet', marker='o', s= 1,
                    transform=cp.crs.PlateCarree())

    # adding colorbar 
    plt.colorbar(sc, ax=ax, orientation='vertical')
    
    plt.show()


# started on a contour map, but it bleeds through the map making it currently unusable
 # def contourmap:   
    # # contourmap:
    # ice_thickness = np.ma.masked_invalid(ice_thickness)
    # ice_mask = ice_thickness <= 0 
    # ice_thickness = np.ma.masked_array(ice_thickness, mask = ice_mask )
    
    
    # levels = np.linspace(np.min(ice_thickness), np.max(ice_thickness), 20)
    # contour = ax.contourf(longitude, latitude, ice_thickness,
    #                       levels=levels, cmap='jet', transform=cp.crs.PlateCarree())
    
    # cbar = plt.colorbar(contour, ax=ax)


# run functions:
# color_plot()
scatter_plot()

