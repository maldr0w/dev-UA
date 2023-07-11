
import matplotlib.pyplot as plt
import numpy as np
import cartopy as cp
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from load_dataset import latitude, longitude, ice_thickness

# plotting functions:

def plot(): # pseudocolor plot 2D
    
        lat_grid = np.arange(latitude.shape[0])
        lon_grid = np.arange(longitude.shape[0])
        
        plt.figure(figsize=(10,6))
        plt.pcolormesh(lon_grid, lat_grid, ice_thickness, cmap='jet')
        plt.colorbar(label='Ice thickness (m)')
        plt.gca().invert_yaxis()
        plt.xlabel('Longitude (degrees east)')
        plt.ylabel('Latitude (degrees north)')
        plt.title('Arctic Region Sea Ice')
        plt.grid(True)
        
        plt.show()  


def scatter_plot(): # Creating map of north pole
    
    fig, ax = plt.subplots(1,1,figsize=(8,8), 
                           subplot_kw={'projection':cp.crs.NorthPolarStereo()})
    
    ax.add_feature(cp.feature.OCEAN)
    ax.add_feature(cp.feature.LAND, edgecolor='black', zorder=1)
    ax.set_facecolor((1.0,1.0,1.0))
    ax.set_extent([-180,180,90,66], cp.crs.PlateCarree())
    
    ax.gridlines(draw_labels=True, alpha=0.5,color='gray', 
                 linestyle='-', linewidth=0.5, 
                 xlocs=np.arange(-180, 181, 30),
                 ylocs=np.arange(60, 91, 5))

    sc = ax.scatter(longitude, latitude, c=ice_thickness,
                    cmap='jet', marker='o', s= 1,
                    transform=cp.crs.PlateCarree())
    
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
    
    # # contourmap:
    # ice_thickness = np.ma.masked_invalid(ice_thickness)
    # ice_mask = ice_thickness <= 0 
    # ice_thickness = np.ma.masked_array(ice_thickness, mask = ice_mask )
    
    
    # levels = np.linspace(np.min(ice_thickness), np.max(ice_thickness), 20)
    # contour = ax.contourf(longitude, latitude, ice_thickness,
    #                       levels=levels, cmap='jet', transform=cp.crs.PlateCarree())
    
    # cbar = plt.colorbar(contour, ax=ax)
    
    plt.show()



# run functions:
# plot()
scatter_plot()

