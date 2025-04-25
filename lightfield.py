import numpy as np
import plotly.express as px
import tifffile

from skimage import morphology
from skimage.morphology import disk
from skimage.measure import label, regionprops

CALIBRATION_THRESHOLD = 20

# In OpticStudio, the Hexagonal Lenslet Array is defined by
# two parameters: # Columns and # Rows (odd positive integers).
# An example of such arrays are shown below for # Columns = 5 and
# # Rows = 3 (left), and # Columns = 3 and # Rows = 3 (right).
# The actual object has a rectangular (almost square) footprint,
# which tightly encompasses all the lenslets, i.e. there are
# partial lenslets at the edges of the object.
#      __    __    __          __
#     /  \__/  \__/  \      __/  \__
#     \__/  \__/  \__/     /  \__/  \
#     /  \__/  \__/  \     \__/  \__/
#     \__/  \__/  \__/     /  \__/  \
#     /  \__/  \__/  \     \__/  \__/
#     \__/  \__/  \__/        \__/
#
HEXAGONAL_LENSLET_ARRAY_COLUMNS = 51
HEXAGONAL_LENSLET_ARRAY_ROWS = 51

if (HEXAGONAL_LENSLET_ARRAY_COLUMNS-1)/2 % 2 == 0:
    TOTAL_LENSLETS = (HEXAGONAL_LENSLET_ARRAY_COLUMNS//2+1)*HEXAGONAL_LENSLET_ARRAY_ROWS
    TOTAL_LENSLETS += (HEXAGONAL_LENSLET_ARRAY_COLUMNS//2+2)*(HEXAGONAL_LENSLET_ARRAY_ROWS+1)
else:
    TOTAL_LENSLETS = (HEXAGONAL_LENSLET_ARRAY_COLUMNS//2+2)*HEXAGONAL_LENSLET_ARRAY_ROWS
    TOTAL_LENSLETS += (HEXAGONAL_LENSLET_ARRAY_COLUMNS//2+1)*(HEXAGONAL_LENSLET_ARRAY_ROWS+1)

def show_image(img, points=None, point_color='rgba(230, 159, 0, 0.5)', point_size=18):
    fig = px.imshow(img, aspect='equal', color_continuous_scale='gray')

    if points is not None:
            fig.add_scatter(
                x=points[:, 1],
                y=points[:, 0],
                mode='markers',
                marker=dict(color=point_color,
                            size=point_size,
                            symbol='circle-open',
                            sizemode='diameter'
                            )
            )

    fig.show()

if __name__ == '__main__':
    raw_data_path = 'light-field_demo.tif'
    calibration_data_path = 'mla_calibration.tif'
    
    raw_data = tifffile.imread(raw_data_path)
    calibration_data = tifffile.imread(calibration_data_path)

    # ===== Find Lenslet Centroids
    binary_calibration = calibration_data > CALIBRATION_THRESHOLD
    opened_calibration = morphology.binary_dilation(binary_calibration, disk(5))
    labeled_calibration = label(opened_calibration)
    centroids = np.array([region.centroid_weighted for region in regionprops(labeled_calibration, intensity_image=calibration_data)])

    # print('Expected number of lenslets =', TOTAL_LENSLETS)
    # print('Found number of lenslets =', len(centroids))
    # show_image(raw_data, points=centroids)

    # ===== Create Light-Field Array
    a_random_centroid_id = 960
    exact_centroid = -centroids[a_random_centroid_id].astype(int)+centroids[a_random_centroid_id]+[20, 20]
    distance_from_centroid = -centroids[a_random_centroid_id].astype(int)+centroids[a_random_centroid_id]
    print(distance_from_centroid)
    single_uv = raw_data[int(centroids[a_random_centroid_id][0])-20:int(centroids[a_random_centroid_id][0])+21, int(centroids[a_random_centroid_id][1])-20:int(centroids[a_random_centroid_id][1])+21]
    show_image(single_uv, points=np.array([[20, 20], exact_centroid]))
    interpolated_uv = np.zeros_like(single_uv)
    old_u = np.arange(-20, 21)
    old_v = np.arange(-20, 21)
    new_u = old_u + distance_from_centroid[0]
    new_v = old_v + distance_from_centroid[1]
    new_u_grid, new_v_grid = np.meshgrid(new_u, new_v)
    points = (old_u, old_v)

    from scipy.interpolate import interpn
    print(single_uv[21, 21])
    show_image(interpn(points, single_uv, (new_v_grid, new_u_grid), bounds_error=False, fill_value=None))
