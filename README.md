# BilevelFilterLearningForImageRecon

This repository contains code to reproduce the figures and demonstrations  
from the paper:

Caroline Crockett and Jeffrey A. Fessler (2022), "Bilevel Methods for Image
Reconstruction", Foundations and Trends® in Signal Processing: Vol. 15: No. 2-3,
pp 121-289. http://dx.doi.org.proxy.lib.umich.edu/10.1561/2000000111

The code is not currently set-up as a package, so the easiest way to use it is
to git clone the repository
``git clone https://github.com/cecroc/BilevelFilterLearningForImageRecon.git``
then run the desired script file from a julia prompt.

Please note the code is likely to undergo continued improvements.

To get started with julia, see the documentation here:
  https://github.com/JeffFessler/MIRT.jl/blob/main/doc/start.md

Description of the available scripts:

- scripts/vertbars.jl: reproduces the demonstrations using the vertical bar
  image (see Appendix D.1)

(the next two scripts are coming soon)

- scripts/cameraman.jl: reproduces the demonstrations using the cameraman image
  (see Appendix D.2). Note the code has the number of iterations set much lower
  than in the paper so that the code runs faster.

- scripts/foundations_and_trends_reproduce: reproduces all of the remaining
  figures in the paper.
