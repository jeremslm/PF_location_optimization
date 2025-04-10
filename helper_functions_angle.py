import numpy as np

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


def place_points(npoints, arc, pol_angles):
    """
    Places coils along a 2D curve (arc) based on given poloidal angles.

    Parameters: - npoints: number of points
                - arc: 2D curve ((R,Z) coordinates)
                - poloidal_angles: List of poloidal angles (in degrees) specifying coil positions

    return:     - locs: 2D array containing the coil centers (as (R,Z) coord)
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
    target_positions = np.interp(pol_angles, theta_range, arclength) # Finds the arc length corresponding to each poloidal angle

    inds = []
    locs = []
    for target in target_positions:
        idx = np.argmin(np.abs(arclength - target)) # index where the value is closest to target
        inds.append(idx)
        locs.append(arc[idx]) # (R,Z) coord of the considered target

    return np.array(inds), np.array(locs)

def update_boundary(r0, z0, a0, kappa, delta, squar, npts=20):
    thp = np.linspace(0,2*np.pi,npts+1)
    thp = thp[:-1]

    ra = r0 + a0*np.cos(thp + delta*np.sin(thp) - squar*np.sin(2*thp))
    za = z0 + kappa*a0*np.sin(thp + squar*np.sin(2*thp))
    return np.vstack([ra, za]).transpose()

def plot_coil(pts, ax, c='k', ls='-', alpha=1):
    ax.plot(np.hstack((pts[:,0],pts[0,0])), np.hstack((pts[:,1],pts[0,1])), c=c, ls=ls, alpha=alpha)