##### ASCEArt1_Variogram #####
# Compute Variogram From Experimental Data
This Python code calculates and plots experimental variograms obtained on a plane, x.
The input data includes:
x: X-coordinates (1D list or array)
y: Y-coordinates (1D list or array)
z: Variable values (if on a grid, z should have the same dimensions
as produced by meshgrid(x,y)).
The code can compute both structured and unstructured experimental variograms
in various directions. You need to specify the lag length and tolerance, the
number of directions, and the initial direction (indicated in degrees; for example, 0°).
