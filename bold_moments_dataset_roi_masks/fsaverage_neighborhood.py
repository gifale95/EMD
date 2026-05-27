from nilearn import datasets
import nibabel as nib
from scipy.spatial import KDTree

# Step 1: Fetch fsaverage surface (full resolution or fsaverage5)
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')  # or 'fsaverage5' for faster computation

# Step 2: Load the inflated surface (left hemisphere)
surf = nib.load(fsaverage['infl_left'])

# Step 3: Extract coordinates and build KDTree
vertices = surf.darrays[0].data  # shape: (n_vertices, 3)
tree = KDTree(vertices) # The KDTree uses 3D Euclidean distances in the surface geometry (not geodesic distances).

# Step 4: Define function to get N nearest neighbors for a vertex
def get_vertex_neighbors(vertex_index, N):
    """Return indices of the given vertex and its N nearest neighbors."""
    distances, indices = tree.query(vertices[vertex_index], k=N+1)
    return indices  # includes the original vertex

# Example usage
vertex_of_interest = 163841 # Index of the target vertex of interest
N = 100 # Number of neughborhood vertices (including the target vertex of interest)
neighbors = get_vertex_neighbors(vertex_of_interest, N)
print(f"N={N} nearest neighbors of vertex {vertex_of_interest}: {neighbors}")
