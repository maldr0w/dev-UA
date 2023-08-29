
import matplotlib.pyplot as plt
import cartopy as cp
import numpy as np

def scatter_plot(longitude,latitude, ice_thickness_nan):
    fig, ax = plt.subplots(1, 1, figsize=(5,5),subplot_kw = {'projection':cp.crs.NorthPolarStereo()})
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
    sc = ax.scatter(latitude,longitude, c= ice_thickness_nan,
                    cmap='jet', marker='o', s= 1,
                    transform=cp.crs.PlateCarree())
    # adding colorbar 
    plt.colorbar(sc, ax=ax, orientation='vertical')

    # remove axis
    # ax.axis('off')    
    
    plt.show()

def pixel_plot(grid, point=None):
    plt.figure(figsize=(5,5))
    plt.imshow(grid, cmap='jet')

    if point is not None:
        x,y = point
        plt.scatter(x, y, color='red')
    
    plt.colorbar(label='Ice Thickness')
    plt.title('Pixel map of sea ice')
    # plt.xlim(150,375)
    # plt.ylim(375,150)
    
    # remove axis
    ax = plt.gca()
    # ax.axis('off')
    # ax.invert_yaxis()    
    plt.show()