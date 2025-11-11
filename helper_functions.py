import numpy as np
from scipy.ndimage import uniform_filter1d

def resize_polygon(points, dx):
    """
    Parameters: - points: limiter (2D array)
                - dx: distance between limiter and coils

    return: - new_points: possible locations for coils
    """
    new_points = np.empty(np.shape(points))
    for i in range(np.shape(points)[0]):
        if i==0:
            last = points[-1,:]
            next = points[i+1,:]
        elif i == np.shape(points)[0]-1:
            last = points[i-1,:]
            next = points[0,:]
        else:
            next = points[i+1,:]
            last = points[i-1,:]
        par = points[i,:]-last
        par/= np.linalg.norm(par)
        perp = np.array([par[1], -par[0]])
        temp = points[i,:] + perp*dx
        par_2 = next-points[i,:]
        par_2/= np.linalg.norm(par_2)
        perp_2 = [par_2[1], -par_2[0]]
        new_points[i, :] = temp + dx/np.dot(perp_2,par)*par + par*dx/np.dot(par_2,perp)*np.dot(par_2,par)
    return new_points


def resize_polygon_MANTA(points, dx, epsilon=1e-10):
    """
    Computes a new polygon by offsetting the input points outward by a distance dx
    (for MANTA limiter)
    
    Parameters:
        - points: (N, 2) array of (R,Z) limiter coordinates
        - dx: Distance between limiter and coils
        - epsilon: Prevents division by zero
    
    Returns:
        - new_points: (N, 2) array of offset polygon points
    """
    N = len(points)
    new_points = np.empty(np.shape(points))
    
    for i in range(np.shape(points)[0]):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[(i + 1) % N]
        
        # Tangent vectors 
        t1 = p_curr - p_prev
        t1 /= np.linalg.norm(t1) + epsilon
        t2 = p_next - p_curr
        t2 /= np.linalg.norm(t2) + epsilon
        
        # Outward normals (rotate tangent vectors 90° clockwise)
        n1 = np.array([t1[1], -t1[0]])
        n2 = np.array([t2[1], -t2[0]])
        
        # Average normal
        n = (n1 + n2)
        n_norm = np.linalg.norm(n)
        if n_norm < epsilon:
            n = n1 
        else:
            n /= n_norm
        
        # Offset point
        new_points[i] = p_curr + dx * n
    
    return new_points

def place_points(npoints, arc, pol_angles):
    """
    Places coils along a 2D curve (arc) based on given poloidal angles.

    Parameters: 
        - npoints: number of points
        - arc: 2D curve ((R,Z) coordinates)
        - poloidal_angles: List of poloidal angles (in degrees) specifying coil positions

    return:     
        - locs: 2D array containing the coil centers (as (R,Z) coord)
        - inds: index of each coil along the arc
    """
    #print("nb of pts given to place_points", npoints)
    if len(pol_angles) != npoints:
        raise ValueError('Warning! poloidal angle distribution length does not match with the number of coils! Overwritting command!')

    # Compute the cumulative arc length along a curve
    arclength = np.zeros(np.size(arc[:,0]))
    for i,point in enumerate(arc):
        if i==0:
            arclength[i] = 0
        else:
            arclength[i] = arclength[i-1] + ((arc[i,0]-arc[i-1,0])**2+(arc[i,1]-arc[i-1,1])**2)**0.5
    totlength = arclength[-1]
   
    # Convert pol. angles to relative positions along the arc
    theta_range = np.linspace(0, 180, len(arc))
    pol_angles = np.sort(pol_angles)   # Ensure angles are sorted
    #target_positions = np.interp(pol_angles, theta_range, arclength) # Finds the arc length corresponding to each poloidal angle

    inds = []
    locs = []
    for target_angle in pol_angles:
        R_tmp = np.interp(target_angle, theta_range, arc[:,0]) # Finds the R position corresponding to each poloidal angle
        Z_tmp = np.interp(target_angle, theta_range, arc[:,1]) # Finds the Z position corresponding to each poloidal angle
        locs.append([R_tmp, Z_tmp])
    
    
    return np.array(inds), np.array(locs)

def update_boundary(r0, z0, a0, kappa, delta, squar, npts=20):
    thp = np.linspace(0,2*np.pi,npts+1)
    thp = thp[:-1]

    ra = r0 + a0*np.cos(thp + delta*np.sin(thp) - squar*np.sin(2*thp))
    za = z0 + kappa*a0*np.sin(thp + squar*np.sin(2*thp))
    return np.vstack([ra, za]).transpose()

def plot_coil(pts, ax, c='k', ls='-', alpha=1):
    ax.plot(np.hstack((pts[:,0],pts[0,0])), np.hstack((pts[:,1],pts[0,1])), c=c, ls=ls, alpha=alpha)

def smoothen(curve, window):
    if window < 1:
        raise ValueError("window must be >=1")
    if window>len(curve):
        raise ValueError(f"window ({window}) cannot exceed curve length ({len(curve)})")
    return uniform_filter1d(curve, window, axis=0, mode='wrap')
