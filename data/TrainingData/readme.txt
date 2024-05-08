NN input params:

  Bo: 1.30500; 
  Rock: 8.10E-06; 
  Po: 5863.8; 
  water viso: 0.31; 
  oil viso: 0.29


2D case input params:

  Producer BHP const: 500 psi; 
  Injector rate const: 500 - 2000 bbl/day
  timestep: 3623


Output_data format:

  WPR and OPR: 3623*13 
  13 well order: I1-I4, P1-P9


Training Dataset:
   
  injrate.mat: 50*4 (50 samples and 4  injs)
  OPR.mat: 3623*13*50 (50 samples of 4 injs and 9 prods)
  WPR.mat: 3623*13*50 (50 samples of 4 injs and 9 prods)


Testing Dataset:

  4 cases:
  500 is I1-I4: 500
  1000 is I1-I4: 1000
  1000_500 is I1-I4: 1000/500/1000/500
  2000 is I1-I4: 2000