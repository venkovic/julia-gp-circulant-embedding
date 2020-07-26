using LinearAlgebra: norm2
using FFTW

const T = Float64

function cov(x::T, y::T, L::T)
  return exp(- (x  -y)^2 / L)
end

struct ToeplitzCov
  dim::Int
  n::Int
  alpha::Vector{ComplexF64}
end

function TCov(Ke1::Vector{T}, dim::Int)
  n = length(Ke1)
  #
  Ce1 = Vector{T}(undef, 2 * (n - 1))
  Ce1[1:n] = Ke1
  for j in 2:(n-1)
    Ce1[2 * (n - 1) - j + 2] = Ce1[j]
  end
  #
  alpha = FFTW.rfft(Ce1)
  return ToeplitzCov(dim, n, alpha)
end

function K_matvec(K::ToeplitzCov, q::Vector{T})
  y = K.alpha .* FFTW.rfft(q)
  q = FFTW.irfft(y, 2 * (K.n - 1))
  return q[1:K.n]
end

import Base: *
(*)(K::ToeplitzCov, x::Vector{T}) = K_matvec(K, x)

function test_1d(n::Int, L::T)
  dx = 1 / (n - 1)
  x = Vector{T}(undef, n)
  for i in 1:n
    x[i] = (i - 1) * dx
  end
  K = Array{T}(undef, n, n)
  for i in 1:n
    for j in 1:n
      K[i, j] = cov(x[i], x[j], L)
    end
  end
  Ke1 = K[:, 1]
  q = zeros(T, 2 * (n - 1))
  q[1:n] = rand(T, n)
  Kq = K * q[1:n]
  Kmat = TCov(Ke1, 1)
  Kq2 = Kmat * q
  println("\n")
  println("Regulat matvec:")
  println(Kq[1], ", ", Kq[2], ", ", Kq[3], ", ...")
  println("FFT-based matvec:")
  println(Kq2[1], ", ", Kq2[2], ", ", Kq2[3], ", ...")
  Kq .-= Kq2
  println("Error norm:")
  println(norm2(Kq))
end

const n = 10
const L = .1
test_1d(n, L)


# module : MyToeplitzCovariances.jl
# Data structure of covariance matrices for stationary Gaussian processes on
# structured grids. Univariate 2D, 3D, (nD?), multivariate 1D. 
# Enables:
#  - fast covariance matrix-vector products.
#  - and other?

# To do: 1. Compare memory use and complexity
#        2. 2D (i.e. BTTB matrices)
#           See 7.2 (p. 276--293) in Lord (2012)
#           Especially Proposition 7.29
