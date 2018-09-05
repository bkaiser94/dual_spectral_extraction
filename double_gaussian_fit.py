"""
Written by Ben Kaiser (UNC-Chapel Hill) 2018-09-05

Take some set of data in one-dimension and fit two gaussians to it with some minimum and maximum allowed separations. they also must have the same sigma value, and must remain within some sort of boundaries for the positions. The sigma values can't get too large since this will eventually be used for the seeing, and that can't get too bad without closing the dome. Sigma will also have to be at least the resolution of the telescope I would think since we can't really say anything about sources we can't resolve.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciop


#constants that should be coming from a header in the future
ccd_height= 200
n_counts = 200
### Constants to be recalibrated based on determinations by user
#sigma_bounds = [1, 20] #update the max sigma value to reflect a seeing of around 3".
sigma_bounds= [0.5,4.24664]
r_bounds = [sigma_bounds[0], ccd_height/3]
#r_bounds = [sigma_bounds[0], 10]

a_bounds= [100, 50000] #flat to saturation-ish values; these are the same for both amplitudes
c_bounds = [-20, 50000] #constant background level in case subtraction is bad
mu_bounds = [80, 120] #location bounds for the lower spectrum trace; this should eventually be a clicked on thing or user input each time.

bound_list = [a_bounds, a_bounds, mu_bounds, r_bounds, sigma_bounds, c_bounds]
bound_array= np.array(bound_list).T
bound_tuple= (bound_array[0], bound_array[1])
print "bound_tuple", bound_tuple
bounds= bound_tuple
def double_gauss(y,a1, a2, mu, r, sigma, c):
    """
    The double_gaussian that has them offset. a1 and mu correspond to the lower gaussian's amplitude and median, respectively, and a2, r correspond to the amplitude and offset of the upper gaussian's median from the first gaussian's median, respectively. sigma applies to both gaussians, and c is a constant for the background.
    
    """
    return (1./(sigma*np.sqrt(2*np.pi)))*(a1*np.exp(-(y-mu)**2/(2*sigma**2))+a2*np.exp(-(y-mu-r)**2/(2*sigma**2)))



def generate_data(y_vals,a1, a2, mu, r, sigma, c):
    """
    Produce randomized versions of the double-gaussians at the locations with the whole ccd column rendered
    
    """
    gauss1_dist= np.random.normal(loc=mu, scale= sigma, size = a1*100)
    gauss2_dist= np.random.normal(loc=mu+r, scale= sigma, size = a2*100)
    #bkg= np.random.normal(loc= y_vals, scale= np.sqrt(c), size= (y_vals.shape[0]))
    bkg= y_vals
    return np.hstack([gauss1_dist, gauss2_dist, bkg])

def generate_binned_data(y_vals,a1, a2, mu, r, sigma, c):
    dataset= generate_data(y_vals,a1, a2, mu, r, sigma, c)
    return np.histogram(dataset, bins= np.append(y_vals,201))

def fit_double_gaussian(y_vals, flux_vals, p0= None):
    print p0
    popt, pcov= sciop.curve_fit(double_gauss, y_vals, flux_vals, bounds= bounds, p0=p0)
    return popt


p0=[40., 40., 85., 5., 2., 1.]
y_vals = np.arange(0,200,1)
dataset= generate_data(y_vals, 40., 40., 85., 10., 2., 1.)
binned_dataset, bin_edges =  generate_binned_data(y_vals, 40., 40., 85., 10., 2., 1.)

param_letters = ['a1', 'a2', 'mu', 'r', 'sigma', 'c']
fit_parameters = fit_double_gaussian(y_vals, binned_dataset)
f_vals = double_gauss(y_vals, fit_parameters[0], fit_parameters[1], fit_parameters[2], fit_parameters[3], fit_parameters[4], fit_parameters[5])
for letter, value in zip(param_letters, fit_parameters):
    print letter, value

plt.plot(y_vals, f_vals, color = 'r')
plt.bar(y_vals, binned_dataset)
plt.show()


plt.plot(np.linspace(0,200, 1000), double_gauss(np.linspace(0,200,1000), fit_parameters[0], fit_parameters[1], fit_parameters[2], fit_parameters[3], fit_parameters[4], fit_parameters[5]), color = 'r')
plt.bar(y_vals, binned_dataset)
plt.show()

#y_vals = np.linspace(0,200, 200)


f_vals = double_gauss(y_vals, 40., 50., 85., 5., 2., 1.)
plt.plot(y_vals, f_vals)
plt.hist(dataset, bins=np.append(y_vals,201))
plt.show()

print binned_dataset.shape
print bin_edges.shape
f_vals = double_gauss(y_vals, 40., 50., 85., 5., 2., 1.)
plt.plot(y_vals, f_vals)
plt.bar(y_vals, binned_dataset)
plt.show()
