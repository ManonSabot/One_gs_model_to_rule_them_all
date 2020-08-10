# libraries
from mpl_toolkits.basemap import Basemap
import pandas as pd
import matplotlib.pyplot as plt
 
# Make a data frame with the GPS of a few cities:
data = pd.DataFrame({'lat':[-19.1833333333, -32.32, -33.61, 34.3833333333,
                            43.48333333, 43.73333333, 44.21666667],
                     'lon':[145.75, 117.87, 150.74, -106.5166666667, 3.75,
                            3.583333333, 4.133333333]})
 
# A basic map
m = Basemap(llcrnrlon=-160, llcrnrlat=-90, urcrnrlon=160, urcrnrlat=90)
m.fillcontinents(color='#f6e8c3')
m.drawcoastlines(linewidth=0.1, color='white')
 
# Add a marker per loc!
x, y = m(data['lon'].values, data['lat'].values)
m.plot(x, y, linestyle='none', marker='o', markersize=5, alpha=0.6,
       c='darkgreen', markeredgecolor='k', markeredgewidth=0.25)

plt.savefig('site_locs.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.close()
