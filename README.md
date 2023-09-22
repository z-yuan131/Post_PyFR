# Post-processing code for PyFR

Please note that these codes are specially designed for aerofoil simulations with PyFR. For the most of functions are deeply modified and need some time to be able to use by the others. Available functions will be showed in this file.


[func-spanavg]                    ; Average the region near a boundary. I.e. wall.
layers = 40                       ; Number of layers.
mode = soln                       ; mesh, soln.
tol = 1e-3                        ; Tolerance for mesh sorting.
nfft = 20                         ; Number of Fourier modes in span
etype = (hex)                     ; Type of elements
bname = (wall)                    ; Attached to some boundaries
box = [(-110, -40), (110, 40)]    ; Limit the region


[func-probes]                     ; Put (several) probe(s) into the field and collect data
exactloc = True                   ; If get exact location of sampling points, if false, return the closest points
format = primitive                ; output format, primitive or conservative
mode = mesh                       ; mesh, soln
porder = 3                        ; polynomial order for new mesh if exists
new-mesh-dir = /path/to/new/mesh  ; If mesh file is ended with .pyfrm, corresponding .pyfrs will be created.
                                  ; If mesh file is not ended with .pyfrm, this file should be h5py file and contain 'mesh' key inside.

[func-gradient]                   ; Calculate gradient inside each element
etype = (hex)                     ; Type of elements
bname = (wall)                    ; Name of boundaries
blsupport = True                  ; If True, values attached to boundary surfaces will be extracted.

[func-Q-criterion]                ; Calculate Q-criterion
etype = (hex, pri)                ; Type of elements  (hex, pri)
bname = (wall)                    ; Name of boundaries
layers = 35                       ; The number of layers away from the boundary.
