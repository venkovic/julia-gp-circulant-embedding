## MyToeplitzCovariance.jl

##### Julia code to enable fast and low storage matrix-vector products with covariance matrices of stationary Gaussian processes.

Author: Nicolas Venkovic.

email: venkovic@gmail.com.



#### Dependencies:

 - Julia 1.4.
 - Julia packages: LinearAlgebra.jl, FFTW.jl. 



#### Explanations: 

__Minimal circulant embedding of covariance matrices of 1D stationary processes__

In 1D, this *.



__Minimal circulant embedding of covariance matrices of 2D stationary processes__

Let a zero mean 2D Gaussian process <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> be sampled at the sites <img src="/tex/c3b1967f5c997becb32d7cf477a50562.svg?invert_in_darkmode&sanitize=true" align=middle width=135.0188169pt height=26.76175259999998pt/> located on a regular <img src="/tex/8769360a81bb4d3d8c7d04b73e4b07c5.svg?invert_in_darkmode&sanitize=true" align=middle width=53.751950999999984pt height=19.1781018pt/> grid ordered from bottom to top and from left to right. That is, <img src="/tex/aba1e10ae977c5735d0dea718ae8a66c.svg?invert_in_darkmode&sanitize=true" align=middle width=16.529662049999992pt height=14.611878600000017pt/> is at the lower left of the grid, <img src="/tex/897d6383d7291d816e6972944fd8654b.svg?invert_in_darkmode&sanitize=true" align=middle width=23.69683964999999pt height=14.611878600000017pt/> is at the upper-left and <img src="/tex/61f42783fe1bb719d798f525b5c49941.svg?invert_in_darkmode&sanitize=true" align=middle width=38.238476099999986pt height=14.611878600000017pt/> is at the upper-right. Let the covariance matrix <img src="/tex/7fe4b22dba654b749d04e5facbd99624.svg?invert_in_darkmode&sanitize=true" align=middle width=114.39925034999997pt height=26.17730939999998pt/> be defined by the components <img src="/tex/5039ed98d08eae6638e4fcef55a30fa5.svg?invert_in_darkmode&sanitize=true" align=middle width=113.54245979999997pt height=24.65753399999998pt/> for a given covariance function <img src="/tex/c8a509ce6f1ce1431a62212cbde866c9.svg?invert_in_darkmode&sanitize=true" align=middle width=118.80103289999998pt height=26.76175259999998pt/> such that <img src="/tex/9db916e88ea3551091dee5d097fcf68f.svg?invert_in_darkmode&sanitize=true" align=middle width=181.08825405pt height=24.65753399999998pt/>. Under these conditions, if <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is stationary, i.e., <img src="/tex/e4c82df656abb48c83b5037cbd72631f.svg?invert_in_darkmode&sanitize=true" align=middle width=177.67053479999998pt height=24.65753399999998pt/> for all <img src="/tex/2f19763355d57a3bfdab353f99ca03cc.svg?invert_in_darkmode&sanitize=true" align=middle width=81.74621235pt height=26.76175259999998pt/>, then the covariance matrix <img src="/tex/558e1b6b0d61666c16dd87622253a301.svg?invert_in_darkmode&sanitize=true" align=middle width=14.817277199999989pt height=22.55708729999998pt/> is block symmetric Toeplitz with symmetric Toeplitz blocks (BSTSTB), and can be recast in the form 



<p align="center"><img src="/tex/42b759730508c6ce4f003e6c273c91cd.svg?invert_in_darkmode&sanitize=true" align=middle width=406.57586145pt height=118.35734295pt/></p>



in which <img src="/tex/72de21135fb91beaec6ac3b4a3ce18cd.svg?invert_in_darkmode&sanitize=true" align=middle width=61.74207599999999pt height=22.55708729999998pt/> are <img src="/tex/3c7e3568fa1625fede3ff436bfec732d.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41942389999999pt height=14.15524440000002pt/>-dimensional symmetric Toeplitz matrices given by



<p align="center"><img src="/tex/680d564a749ef86663916f4a0e0454d4.svg?invert_in_darkmode&sanitize=true" align=middle width=736.38334275pt height=118.35734295pt/></p>



Every Toeplitz block <img src="/tex/72de21135fb91beaec6ac3b4a3ce18cd.svg?invert_in_darkmode&sanitize=true" align=middle width=61.74207599999999pt height=22.55708729999998pt/> is minimally embedded into an <img src="/tex/4566ad2ef4d92d1bf1bdc04bf8c30c61.svg?invert_in_darkmode&sanitize=true" align=middle width=66.55637999999999pt height=24.65753399999998pt/>-dimensional symmetric circulant matrix <img src="/tex/09192eeb0aba91d05ffd641fb78da491.svg?invert_in_darkmode&sanitize=true" align=middle width=19.75740524999999pt height=22.55708729999998pt/> with components 



<p align="center"><img src="/tex/3adc96721f2d7590636790ca9df1f732.svg?invert_in_darkmode&sanitize=true" align=middle width=707.81570475pt height=230.73982965pt/></p>



These blocks can then be used to form a block symmetric circulant matrix with symmetric circulant blocks (BSCSCB) given by



<p align="center"><img src="/tex/a2779f433f4211f90ca7abcfe5715582.svg?invert_in_darkmode&sanitize=true" align=middle width=580.7685509999999pt height=230.73982965pt/></p>



This matrix is <img src="/tex/3057344cdf9b5087fefe046df899b96d.svg?invert_in_darkmode&sanitize=true" align=middle width=124.89355065pt height=24.65753399999998pt/>-dimensional, minimally embeds <img src="/tex/558e1b6b0d61666c16dd87622253a301.svg?invert_in_darkmode&sanitize=true" align=middle width=14.817277199999989pt height=22.55708729999998pt/>, and has the following block structure:



<p align="center"><img src="/tex/177f35e9320705dc6fc5c0f1f36b2ed7.svg?invert_in_darkmode&sanitize=true" align=middle width=844.3764400499999pt height=339.07064399999996pt/></p>



where the blocks <img src="/tex/7c74eeb32158ff7c4f67d191b95450fb.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=15.296829900000011pt/> and <img src="/tex/22fcde5697fb6ff191e860c19adb9cf6.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=14.611911599999981pt/> are <img src="/tex/3c7e3568fa1625fede3ff436bfec732d.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41942389999999pt height=14.15524440000002pt/> and <img src="/tex/3c7e3568fa1625fede3ff436bfec732d.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41942389999999pt height=14.15524440000002pt/>-dimensional, respectively, and can both be inferred from Eqs. (3)--(4).

Let us then re-write <img src="/tex/214a2cc1d1a8bc2d81b64dc1fbc0f7f7.svg?invert_in_darkmode&sanitize=true" align=middle width=70.20179264999999pt height=22.648391699999998pt/> as follows



<p align="center"><img src="/tex/35dca25032276767211a92d5a71561a5.svg?invert_in_darkmode&sanitize=true" align=middle width=196.44088365pt height=88.76800184999999pt/></p>



where every block <img src="/tex/467acb4e59bea4b7da08bea4f1ab116a.svg?invert_in_darkmode&sanitize=true" align=middle width=56.90191484999999pt height=14.611878600000017pt/> is <img src="/tex/3c7e3568fa1625fede3ff436bfec732d.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41942389999999pt height=14.15524440000002pt/>-dimensional. Similarly, we have



<p align="center"><img src="/tex/6ab83d2d2f02012a4fda88c2a41c2ea0.svg?invert_in_darkmode&sanitize=true" align=middle width=530.704086pt height=88.76800184999999pt/></p>



Then, note that we have



<p align="center"><img src="/tex/651282b7522bbe571ba475af2d5e1e88.svg?invert_in_darkmode&sanitize=true" align=middle width=613.5096462pt height=168.46658235pt/></p>



where the blocks <img src="/tex/892427b024f10b6ad997a062853ffa1b.svg?invert_in_darkmode&sanitize=true" align=middle width=39.58741544999999pt height=15.296829900000011pt/> and <img src="/tex/75d90ab5692c8bdd2050fd9b78b61be3.svg?invert_in_darkmode&sanitize=true" align=middle width=98.05623134999998pt height=14.611911599999981pt/> are unnecessary byproducts of the computation of <img src="/tex/592a1fd8399f3ab33f9453e3396d8abe.svg?invert_in_darkmode&sanitize=true" align=middle width=24.79439324999999pt height=22.55708729999998pt/> produced by the matrix-vector product of <img src="/tex/12d3ebda1a212bd89197298f60cf3ce1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.652895299999988pt height=22.55708729999998pt/> with a zero-padding extension of <img src="/tex/e73485aa867794d51ccd8725055d03a3.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/>. Note that a more clever use of memory and computational effort could be achieved by considering alternatives to zero-padding. Either way, the advantage of embedding <img src="/tex/558e1b6b0d61666c16dd87622253a301.svg?invert_in_darkmode&sanitize=true" align=middle width=14.817277199999989pt height=22.55708729999998pt/> lies in the fact that <img src="/tex/12d3ebda1a212bd89197298f60cf3ce1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.652895299999988pt height=22.55708729999998pt/>, as a BCCB matrix, admits the following decomposition (Lord et al., 2014) :



<p align="center"><img src="/tex/4065ab52df47fdc21125d8ac574888c5.svg?invert_in_darkmode&sanitize=true" align=middle width=513.1709979pt height=18.17354385pt/></p>


where the components <img src="/tex/40642dc37e9cf94c12bdd68e52bdfdd4.svg?invert_in_darkmode&sanitize=true" align=middle width=224.24277150000003pt height=34.337843099999986pt/> of the Fourier matrix <img src="/tex/3120000f5688c1807b25cc1239d8737d.svg?invert_in_darkmode&sanitize=true" align=middle width=79.33223265pt height=26.17730939999998pt/> are defined with


<p align="center"><img src="/tex/9589448f638e981117d0e9f3804cde64.svg?invert_in_darkmode&sanitize=true" align=middle width=263.38665375pt height=39.452455349999994pt/></p>


as in Van Loan (1992). In Eq. (9), we have


<p align="center"><img src="/tex/e8596d5889ffebe3d743f976850d518a.svg?invert_in_darkmode&sanitize=true" align=middle width=512.81988615pt height=18.17354385pt/></p>
 

Then, for all <img src="/tex/b3c7d448fba6b4a3029b06b6aea5e689.svg?invert_in_darkmode&sanitize=true" align=middle width=124.89355065pt height=24.65753399999998pt/>-dimensional vectors <img src="/tex/f964f74151485ed723d54cb92c0d7a30.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=22.831056599999986pt/>, we can compute <img src="/tex/502e9be8b959405c1facad8e29b7bff5.svg?invert_in_darkmode&sanitize=true" align=middle width=24.15512384999999pt height=22.831056599999986pt/> by the following procedure;


<p align="center"><img src="/tex/3eb447e5d60d6bf4194b724e11e332b5.svg?invert_in_darkmode&sanitize=true" align=middle width=204.04024245pt height=164.38356165pt/></p>


where <img src="/tex/c0463eeb4772bfde779c20d52901d01b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=14.611911599999981pt/> refers to the Hadamard product.

As a result, for all <img src="/tex/214a2cc1d1a8bc2d81b64dc1fbc0f7f7.svg?invert_in_darkmode&sanitize=true" align=middle width=70.20179264999999pt height=22.648391699999998pt/>, the matrix-vector product <img src="/tex/42d793b53196095b787a5e3c45d2b302.svg?invert_in_darkmode&sanitize=true" align=middle width=24.79439324999999pt height=22.55708729999998pt/> is computed at a cost of O(<img src="/tex/59ebe596a9e26f55cbbb7a89453914b2.svg?invert_in_darkmode&sanitize=true" align=middle width=102.16177619999998pt height=24.65753399999998pt/>) instead of O(<img src="/tex/2cc4fcc186a362cff729530b5f9318eb.svg?invert_in_darkmode&sanitize=true" align=middle width=33.66075899999999pt height=26.76175259999998pt/>), while only using O(<img src="/tex/4b2b6901bd751903eef2b8b5a21b82f6.svg?invert_in_darkmode&sanitize=true" align=middle width=33.66075899999999pt height=14.15524440000002pt/>) of memory to store the essential information about <img src="/tex/558e1b6b0d61666c16dd87622253a301.svg?invert_in_darkmode&sanitize=true" align=middle width=14.817277199999989pt height=22.55708729999998pt/>. 



__Minimal circulant embedding of covariance matrices of multivariate stationary time series__



#### Running examples: 

1D Example

```bash
julia starter.jl
```

2D Example

```bash
julia starter_2d.jl
```



#### References:

- **Anitescu, M.; Chen, J. & Wang, L.** (2012) **A matrix-free approach for solving the parametric Gaussian process maximum likelihood problem**, SIAM Journal on Scientific Computing, SIAM, 34, A240-A262.
- **Lord, G. J.; Powell, C. E. & Shardlow**, T. (2014) **An introduction to computational stochastic PDEs**, Cambridge University Press, 50.  
- **Golub, G. & Van Loan, C.** (2013) **Matrix Computations**, Johns Hopkins University Press.  
- **Van Loan, C.** (1992) **Computational frameworks for the fast Fourier transform**, SIAM.