using LinearAlgebra: norm2, Symmetric, isposdef
using FFTW

const T = Float64

function cov(x::Vector{T}, y::Vector{T}, L::T)
  return exp(- (x  - y)' * (x  - y) / L)
end

struct ToeplitzCov
  dim::Int
  n::Int
  hCred::Array{ComplexF64,2}
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






function test_2d(n1::Int, n2::Int, L::T)
  #
  # Assemble full covariance matrix K
  dx1 = 1 / (n1 - 1)
  x1 = Vector{T}(undef, n1)
  for i in 1:n1
    x1[i] = (i - 1) * dx1
  end
  dx2 = 1 / (n2 - 1)
  x2 = Vector{T}(undef, n2)
  for j in 1:n2
    x2[j] = (j - 1) * dx2
  end
  n = n1 * n2
  Kupper = Array{T}(undef, n, n)
  #y1, y2 = [x1[1], x2[1]], zeros(T, 2)
  #for i in 1:n1
  #  for j in 1:n2
  #    y2 = [x1[i], x2[j]]
  #    Kupper[i, j] = cov(y1, y2, L)
  #  end
  #end
  y1, y2 = zeros(T, 2), zeros(T, 2)
  for k in 1:n
    for l in k:n
      j1 = div(k - 1, n1) + 1
      i1 = k - (j1 - 1) * n1
      y1 = [x1[i1], x2[j1]]
      #
      j2 = div(l - 1, n1) + 1
      i2 = l - (j2 - 1) * n1
      y2 = [x1[i2], x2[j2]]
      #
      Kupper[k, l] = cov(y1, y2, L)
    end
  end
  K = Symmetric(Kupper)
  println()
  println("Is K really BTTB?")
  println(norm2(K[1:n1-1, 1] - K[2:n1, 2]))
  println(norm2(K[1:n1, 1:n1] - K[n1+1:2*n1, n1+1:2*n1]))
  println(norm2(K[1:n1, n1+1:2*n1] - K[n1+1:2*n1, 2*n1+1:3*n1]))
  #
  # Assemble the reduced matrix form of the BCCB matrix which minimally embeds K
  Cred = Array{T}(undef, 2 * (n1 - 1),  2 * (n2 - 1))
  for j in 1:n2
    Cred[1:n1, j] = K[1:n1, (j-1)*n1+1]
    for i in 2:(n1-1)
      Cred[2 * (n1 - 1) - i + 2, j] = Cred[i, j]
    end
    if (1 < j) && (j < n2)
      Cred[:, 2 * (n2 - 1) - j + 2] = Cred[:, j]
    end
  end
  hCred = fft(Cred, 2) #/ sqrt(4 * (n2 - 1) * (n1 - 1))
  println(hCred)
  println(isposdef(Cred))
  println()
  println("hCred should be diagonal:")
  println(hCred[1:4, 1:4])

  #
  q = rand(T, n)
  Kq = K * q
  #
  Q = zeros(T, 2 * (n1 - 1), 2 * (n2 - 1))
  for j in 1:n2
    Q[1:n1, j] = q[(j-1)*n1+1:j*n1]
  end
  Q = fft(Q, 2)
  Q = Q .* hCred
  Q = ifft(Q, 2)
  #
  Kq2 = zeros(T, n)
  for j in 1:n2
    Kq2[(j-1)*n1+1:j*n1] = real(Q[1:n1, j])
  end
  #
  #Kmat = TCov(Ke1, 1)
  #Kq2 = Kmat * q
  println("\n")
  println("Regulat matvec:")
  println(Kq[1], ", ", Kq[2], ", ", Kq[3], ", ...")
  println("FFT-based matvec:")
  println(Kq2[1], ", ", Kq2[2], ", ", Kq2[3], ", ...")
  Kq .-= Kq2
  println("Error norm:")
  println(norm2(Kq))
end

const n1 = 6
const n2 = 9
const L = .1
test_2d(n1, n2, L)


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
