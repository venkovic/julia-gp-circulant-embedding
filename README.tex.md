## MyToeplitzCovariance.jl

##### Julia code of ***.

Goal: Circulant embedding of Toeplitz and BTTB matrices for fast covariance matrix-vector products and simulation.

Author: Nicolas Venkovic.

email: venkovic@gmail.com.



#### Dependencies:

 - Julia 1.4.
 - Julia packages: LinearAlgebra.jl, FFTW.jl. 



####Explanations: 

__Minimal circulant embedding of covariance matrices of 1D stationary processes__

In 1D, this *.

__Minimal circulant embedding of covariance matrices of 2D stationary processes__

We order the data like this *** and define the components of the covariance matrix $\mathbf{K}\in\mathbb{R}^{n_1n_2\times n_1n_2}$ by $K_{ij}:=k(\mathbf{x}_i,\mathbf{x}_j)$ for a given covariance function $k:\mathbb{R}^2\times\mathbb{R}^2\rightarrow\mathbb{R}$. Let us assume that $k$ is *. Under these conditions, if $k$ is stationary, i.e., $k(\mathbf{x},\mathbf{y})=k(\mathbf{x}+\mathbf{z},\mathbf{y}+\mathbf{z})$ for all $\mathbf{x},\mathbf{y},\mathbf{z}\in\mathbb{R}^2$, then the covariance matrix $\mathbf{K}$ is block symmetric Toeplitz with symmetric Toeplitz blocks (BSTSTB), and can be recast in the form 
$$
\mathbf{K}:=
\left[\begin{matrix}
\mathbf{K}_1&\mathbf{K}_2&\cdots&\mathbf{K}_{n_2-1}&\mathbf{K}_{n_2}\\
 &\mathbf{K}_1&\ddots&\ddots&\mathbf{K}_{n_2-1}\\
&&\ddots&\ddots&\vdots\\
&&&\mathbf{K}_{1}&\mathbf{K}_{2}\\
&&&&\mathbf{K}_{1}
\end{matrix}\right]
$$
in which $\mathbf{K}_{1\leq j\leq n_2}$ are $n_1$-dimensional symmetric Toeplitz matrices given by
$$
\mathbf{K}_j=
\left[\begin{matrix}
(K_j)_1&(K_j)_2&\cdots&(K_j)_{n_1-1}&(K_j)_{n_1}\\
&(K_j)_1&\ddots&\ddots&(K_j)_{n_1-1}\\
&&\ddots&\ddots&\vdots\\
&&&(K_j)_1&(K_j)_2\\
&&&&(K_j)_1
\end{matrix}\right]
\;\;\mathrm{where}\;\;
(K_j)_\ell:=k(\mathbf{x}_1,\mathbf{x}_{(j-1)n_1+\ell}).
$$



Every Toeplitz block $\mathbf{K}_{1\leq j\leq n_2}$ is minimally embedded into an $2(n_1-1)$-dimensional symmetric circulant matrix $\mathbf{C}_j$ with components 
$$
\mathbf{C}_j=
\left[\begin{matrix}
(K_j)_{1}&(K_j)_{2}&\cdots&(K_j)_{n_1-1}&(K_j)_{n_1}&\vline&(K_j)_{n_1-1}&\cdots&(K_j)_{2}\\
&(K_j)_{1}&\ddots&\ddots&(K_j)_{n_1-1}&\vline&(K_j)_{n_1}&\cdots&(K_j)_{3}\\
&&\ddots&\ddots&\vdots&\vline&\vdots&&\vdots\\
&&&(K_j)_{1}&(K_j)_{2}&\vline&(K_j)_{3}&\cdots&(K_j)_{n_1}\\
&&&&(K_j)_{1}&\vline&(K_j)_{2}&\cdots&(K_j)_{n_1-1}\\
\\\hline\\
&&&&&\vline&(K_j)_{1}&\cdots&(K_j)_{n_1-2}\\
&&&&&\vline&&\ddots&\vdots\\
&&&&&\vline&&&(K_j)_{1}\\
\end{matrix}\right].
$$


These blocks can then be used to form a block symmetric circulant matrix with symmetric circulant blocks (BSCSCB) given by
$$
\mathbf{C}=
\left[\begin{matrix}
\mathbf{C}_1&\mathbf{C}_2&\cdots&\mathbf{C}_{n_1-1}&\mathbf{C}_{n_1}&\vline&\mathbf{C}_{n_1-1}&\cdots&\mathbf{C}_{2}\\
&\mathbf{C}_1&\ddots&\ddots&\mathbf{C}_{n_1-1}&\vline&\mathbf{C}_{n_1}&\cdots&\mathbf{C}_{3}\\
&&\ddots&\ddots&\vdots&\vline&\vdots&&\vdots\\
&&&\mathbf{C}_{1}&\mathbf{C}_{2}&\vline&\mathbf{C}_{3}&\cdots&\mathbf{C}_{n_1}\\
&&&&\mathbf{C}_{1}&\vline&\mathbf{C}_{2}&\cdots&\mathbf{C}_{n_1-1}\\
\\\hline\\
&&&&&\vline&\mathbf{C}_{1}&\cdots&\mathbf{C}_{n_1-2}\\
&&&&&\vline&&\ddots&\vdots\\
&&&&&\vline&&&\mathbf{C}_{1}\\
\end{matrix}\right].
$$

This matrix is $4(n_2-1)(n_1-1)$-dimensional, minimally embeds $\mathbf{K}$, and has the following block structure:

$$
\mathbf{C}:=
\left[\begin{matrix}
%
% 1st line, C[1:2*(n1-1), :]
\left[\begin{matrix} \mathbf{K}_1&*\\ *&\bullet \end{matrix}\right] &
\left[\begin{matrix} \mathbf{K}_2&*\\ *&\bullet \end{matrix}\right] &
\cdots &
\left[\begin{matrix} \mathbf{K}_{n_2-1}&*\\ *&\bullet \end{matrix}\right] &
\left[\begin{matrix} \mathbf{K}_{n_2}&*\\ *&\bullet \end{matrix}\right]&
\vline&
\left[\begin{matrix} \mathbf{K}_{n_2-1}&*\\ *&\bullet \end{matrix}\right]&
\cdots &
\left[\begin{matrix} \mathbf{K}_{2}&*\\ *&\bullet \end{matrix}\right]\\
%
% 2nd line, C[1:2*(n1-1), :]
&
\left[\begin{matrix} \mathbf{K}_1&*\\ *&\bullet \end{matrix}\right] &
\ddots & 
\ddots &
\left[\begin{matrix} \mathbf{K}_{n_2-1}&*\\ *&\bullet \end{matrix}\right] &
\vline&
\left[\begin{matrix} \mathbf{K}_{n_2}&*\\ *&\bullet \end{matrix}\right] &
\cdots &
\left[\begin{matrix} \mathbf{K}_{3}&*\\ *&\bullet \end{matrix}\right]\\
%
% 3rd line
&&\ddots&\ddots&\vdots&\vline&\vdots&&\vdots\\
%
% 4th line, C[1:2*(n1-1), :]
&&&
\left[\begin{matrix} \mathbf{K}_{1}&*\\ *&\bullet \end{matrix}\right] &
\left[\begin{matrix} \mathbf{K}_{2}&*\\ *&\bullet \end{matrix}\right] &
\vline&
\left[\begin{matrix} \mathbf{K}_{3}&*\\ *&\bullet \end{matrix}\right] &
\cdots &
\left[\begin{matrix} \mathbf{K}_{n_2}&*\\ *&\bullet \end{matrix}\right]\\
%
% 5th line, C[1:2*(n1-1), :]
&&&&
\left[\begin{matrix} \mathbf{K}_{1}&*\\ *&\bullet \end{matrix}\right] &
\vline&
\left[\begin{matrix} \mathbf{K}_{2}&*\\ *&\bullet \end{matrix}\right] &
\cdots &
\left[\begin{matrix} \mathbf{K}_{n_2-1}&*\\ *&\bullet \end{matrix}\right]\\
%
% 6th line, C[1:2*(n1-1), :]
\\\hline\\
&&&&&\vline&
\left[\begin{matrix} \mathbf{K}_{1}&*\\ *&\bullet \end{matrix}\right] &
\cdots &
\left[\begin{matrix} \mathbf{K}_{n_2-2}&*\\ *&\bullet \end{matrix}\right]\\
%
% 7th line
&&&&&\vline&&\ddots &\vdots\\
%
% 8th line, C[1:2*(n1-1), :]
&&&&&\vline&&&
\left[\begin{matrix} \mathbf{K}_{1}&*\\ *&\bullet \end{matrix}\right] \end{matrix}\right].
$$


Let us then re-write $\mathbf{q}\in\mathbb{R}^{n_1n_2}$ as follows
$$
\mathbf{q}=
\left[
\begin{matrix}
\mathbf{q}_1\\
\mathbf{q}_2\\
\vdots\\
\mathbf{q}_{n_2}\\
\end{matrix}
\right]
$$
where every block $\mathbf{q}_{1\leq j\leq n_2}$ is $n_1$-dimensional, so that

$$
\mathbf{Kq}=
\left[
\begin{matrix}
\mathbf{K}_1\mathbf{q}_1+\mathbf{K}_1\mathbf{q}_2+\cdots+\mathbf{K}_{n_2}\mathbf{q}_{n_2}\\
\mathbf{K}_2\mathbf{q}_1+\mathbf{K}_2\mathbf{q}_2+\cdots+\mathbf{K}_{n_2-1}\mathbf{q}_{n_2}\\
\vdots\\
\mathbf{K}_{n_2}\mathbf{q}_1+\mathbf{K}_{n_2-1}\mathbf{q}_2+\cdots+\mathbf{K}_{1}\mathbf{q}_{n_2}\\
\end{matrix}
\right]
=:
\left[
\begin{matrix}
(\mathbf{K}\mathbf{q})_1\\
(\mathbf{K}\mathbf{q})_2\\
\vdots\\
(\mathbf{K}\mathbf{q})_{n_2}\\
\end{matrix}
\right].
$$
Then, note that we have

$$
\mathbf{C}
\left[
\begin{matrix}
\mathbf{q}_1\\
\mathbf{0}_{n_1-2}\\
\mathbf{q}_2\\
\mathbf{0}_{n_1-2}\\
\vdots\\
\mathbf{q}_{n_2}\\
\mathbf{0}_{n_1-2}\\
\hline
\mathbf{0}_{2(n_2-2)(n_1-1)}
\end{matrix}
\right]=
\left[
\begin{matrix}
\mathbf{K}_{1}\mathbf{q}_{1}+\cdots+\mathbf{K}_{n_2}\mathbf{q}_{n_2}\\
\mathbf{*}_{n_1-2}\\
\mathbf{K}_{2}\mathbf{q}_{1}+\cdots+\mathbf{K}_{n_2-1}\mathbf{q}_{n_2}\\
\mathbf{*}_{n_2-2}\\
\vdots\\
\mathbf{K}_{n_2}\mathbf{q}_{1}+\cdots+\mathbf{K}_{1}\mathbf{q}_{n_2}\\
\mathbf{*}_{n_1-2}\\
\hline
\mathbf{*}_{2(n_2-2)(n_1-1)}
\end{matrix}
\right]=
\left[
\begin{matrix}
(\mathbf{K}\mathbf{q})_{1}\\
\mathbf{*}_{n_1-2}\\
(\mathbf{K}\mathbf{q})_{2}\\
\mathbf{*}_{n_2-2}\\
\vdots\\
(\mathbf{K}\mathbf{q})_{n_2}\\
\mathbf{*}_{n_1-2}\\
\hline
\mathbf{*}_{2(n_2-2)(n_1-1)}
\end{matrix}
\right]
$$
so that the components of $\mathbf{Kq}$ can be extracted upon computing the matrix-vector product of $\mathbf{C}$ with an extension of $\mathbf{q}$. Moreover, because $\mathbf{C}$ is BCCB, it admits the following decomposition
$$
\mathbf{C}(\mathbf{F}_{2(n_2-1)}\otimes\mathbf{F}_{2(n_2-1)})=
\boldsymbol{\Lambda}(\mathbf{F}_{2(n_2-1)}\otimes\mathbf{F}_{2(n_2-1)})
$$
where *.

$\mathrm{DFT}_2[\mathbf{x}]=$

 

As a result, for all $\mathbf{q}\in\mathbb{R}^{n_1n_2}$, the matrix-vector product $\mathbf{K}\mathbf{q}$ is computed at a cost of O($\log(n_1n_2)n_1n_2$) instead of O($n_1^2n_2^2$), while only using O($n_1n_2$) of memory to store the essential information about $\mathbf{K}$. 



__Minimal circulant embedding of covariance matrices of multivariate stationary time series__



#### Running examples: 





#### References:

- **Anitescu, M.; Chen, J. & Wang, L.** (2012) **A matrix-free approach for solving the parametric Gaussian process maximum likelihood problem**, SIAM Journal on Scientific Computing, SIAM, 34, A240-A262.
- **Lord, G. J.; Powell, C. E. & Shardlow**, T. (2014) **An introduction to computational stochastic PDEs**, Cambridge University Press, 50.  
- **Golub, G. & Van Loan, C.** (2013) **Matrix Computations**, Johns Hopkins University Press.  
- **Van Loan, C.** (1992) **Computational frameworks for the fast Fourier transform**, SIAM.