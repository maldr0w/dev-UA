
import matplotlib.pyplot as plt
import numpy as np
import cartopy as cp

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig, ax = plt.subplots(1,1,figsize=(8,8), 
            subplot_kw={'projection':cp.crs.NorthPolarStereo()})


ax.set_extent([-180,180,90,66], cp.crs.PlateCarree())
ax.coastlines()
# ax.add_feature(cp.feature.LAND, edgecolor='black', zorder=1)
# ax.legend(fontsize='x-large')

gl = ax.gridlines(draw_labels=True, alpha=0.5,color='gray', 
             linestyle='-', linewidth=0.5, 
             xlocs=np.arange(-180, 181, 30),
             ylocs=np.arange(60, 91, 5))

# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.ylabels_right = True

gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

# gl.xformatter = LONGITUDE_FORMATTER

plt.show()

