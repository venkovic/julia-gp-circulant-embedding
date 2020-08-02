## MyToeplitzCovariance.jl

##### Julia code to enable fast and low storage matrix-vector products with covariance matrices of stationary Gaussian processes.

Author: Nicolas Venkovic.

email: venkovic@gmail.com.



#### Dependencies:

 - Julia 1.4.
 - Julia packages: LinearAlgebra.jl, FFTW.jl. 



####Explanations: 

__Minimal circulant embedding of covariance matrices of 1D stationary processes__

In 1D, this *.



__Minimal circulant embedding of covariance matrices of 2D stationary processes__

We order the data like this *** and define the components of the covariance matrix <img src="/tex/7fe4b22dba654b749d04e5facbd99624.svg?invert_in_darkmode&sanitize=true" align=middle width=114.39925034999997pt height=26.17730939999998pt/> by <img src="/tex/5039ed98d08eae6638e4fcef55a30fa5.svg?invert_in_darkmode&sanitize=true" align=middle width=113.54245979999997pt height=24.65753399999998pt/> for a given covariance function <img src="/tex/c8a509ce6f1ce1431a62212cbde866c9.svg?invert_in_darkmode&sanitize=true" align=middle width=118.80103289999998pt height=26.76175259999998pt/>. Let us assume that <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is *. Under these conditions, if <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is stationary, i.e., <img src="/tex/e4c82df656abb48c83b5037cbd72631f.svg?invert_in_darkmode&sanitize=true" align=middle width=177.67053479999998pt height=24.65753399999998pt/> for all <img src="/tex/2f19763355d57a3bfdab353f99ca03cc.svg?invert_in_darkmode&sanitize=true" align=middle width=81.74621235pt height=26.76175259999998pt/>, then the covariance matrix <img src="/tex/558e1b6b0d61666c16dd87622253a301.svg?invert_in_darkmode&sanitize=true" align=middle width=14.817277199999989pt height=22.55708729999998pt/> is block symmetric Toeplitz with symmetric Toeplitz blocks (BSTSTB), and can be recast in the form 
<p align="center"><img src="/tex/5a74b3b9eabc662ae01ea1b330a0d2a4.svg?invert_in_darkmode&sanitize=true" align=middle width=289.28827125pt height=118.35734295pt/></p>
in which <img src="/tex/72de21135fb91beaec6ac3b4a3ce18cd.svg?invert_in_darkmode&sanitize=true" align=middle width=61.74207599999999pt height=22.55708729999998pt/> are <img src="/tex/3c7e3568fa1625fede3ff436bfec732d.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41942389999999pt height=14.15524440000002pt/>-dimensional symmetric Toeplitz matrices given by
<p align="center"><img src="/tex/259c8e92a45d5583dcfd1476860aef75.svg?invert_in_darkmode&sanitize=true" align=middle width=621.8354092499999pt height=118.35734295pt/></p>



Every Toeplitz block <img src="/tex/72de21135fb91beaec6ac3b4a3ce18cd.svg?invert_in_darkmode&sanitize=true" align=middle width=61.74207599999999pt height=22.55708729999998pt/> is minimally embedded into an <img src="/tex/4566ad2ef4d92d1bf1bdc04bf8c30c61.svg?invert_in_darkmode&sanitize=true" align=middle width=66.55637999999999pt height=24.65753399999998pt/>-dimensional symmetric circulant matrix <img src="/tex/09192eeb0aba91d05ffd641fb78da491.svg?invert_in_darkmode&sanitize=true" align=middle width=19.75740524999999pt height=22.55708729999998pt/> with components 
<p align="center"><img src="/tex/987611952400ebdc841c7377c206d18a.svg?invert_in_darkmode&sanitize=true" align=middle width=593.2677696pt height=230.73982965pt/></p>


These blocks can then be used to form a block symmetric circulant matrix with symmetric circulant blocks (BSCSCB) given by
<p align="center"><img src="/tex/abcce65d578d12166313e4b50b1bb525.svg?invert_in_darkmode&sanitize=true" align=middle width=466.2206175pt height=230.73982965pt/></p>

This matrix is <img src="/tex/3057344cdf9b5087fefe046df899b96d.svg?invert_in_darkmode&sanitize=true" align=middle width=124.89355065pt height=24.65753399999998pt/>-dimensional, minimally embeds <img src="/tex/558e1b6b0d61666c16dd87622253a301.svg?invert_in_darkmode&sanitize=true" align=middle width=14.817277199999989pt height=22.55708729999998pt/>, and has the following block structure:

<p align="center"><img src="/tex/ad54f8897c2f40dbe8bfe0fa51b64f08.svg?invert_in_darkmode&sanitize=true" align=middle width=729.8285065499999pt height=339.07064399999996pt/></p>


Let us then re-write <img src="/tex/214a2cc1d1a8bc2d81b64dc1fbc0f7f7.svg?invert_in_darkmode&sanitize=true" align=middle width=70.20179264999999pt height=22.648391699999998pt/> as follows
<p align="center"><img src="/tex/de1046954ff06ff89cf2334dfd974730.svg?invert_in_darkmode&sanitize=true" align=middle width=79.15329344999999pt height=88.76800184999999pt/></p>
where every block <img src="/tex/467acb4e59bea4b7da08bea4f1ab116a.svg?invert_in_darkmode&sanitize=true" align=middle width=56.90191484999999pt height=14.611878600000017pt/> is <img src="/tex/3c7e3568fa1625fede3ff436bfec732d.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41942389999999pt height=14.15524440000002pt/>-dimensional, so that

<p align="center"><img src="/tex/9eca954c44a029051169704470c1afb2.svg?invert_in_darkmode&sanitize=true" align=middle width=416.15615249999996pt height=88.76800184999999pt/></p>
Then, note that we have

<p align="center"><img src="/tex/4173787a5aee5be9606056957f056b18.svg?invert_in_darkmode&sanitize=true" align=middle width=496.22205599999995pt height=168.46658235pt/></p>
so that the components of <img src="/tex/592a1fd8399f3ab33f9453e3396d8abe.svg?invert_in_darkmode&sanitize=true" align=middle width=24.79439324999999pt height=22.55708729999998pt/> can be extracted upon computing the matrix-vector product of <img src="/tex/12d3ebda1a212bd89197298f60cf3ce1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.652895299999988pt height=22.55708729999998pt/> with an extension of <img src="/tex/e73485aa867794d51ccd8725055d03a3.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/>. Moreover, because <img src="/tex/12d3ebda1a212bd89197298f60cf3ce1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.652895299999988pt height=22.55708729999998pt/> is BCCB, it admits the following decomposition
<p align="center"><img src="/tex/d2ec7b591a53f420197f1c17d85550c4.svg?invert_in_darkmode&sanitize=true" align=middle width=358.21198215pt height=18.17354385pt/></p>
where *.

<img src="/tex/cac7f6e309aaf7702b909e00496bb2f2.svg?invert_in_darkmode&sanitize=true" align=middle width=78.99544289999999pt height=24.65753399999998pt/>

 

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