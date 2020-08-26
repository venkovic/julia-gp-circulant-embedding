using FFTW: rfft, irfft
using LinearAlgebra: dot

"""
bststb_1d(n, e1)

Structure for symmetric n-by-n Toeplitz matrix. Only the first column e1 of the
matrix is stored.
"""
struct bststb_1d
  n::Int
  e1::Vector{T}
end

"""
matvec(Toe, x)

Performs a (naive) matrix-vector product between a symmetric n-by-n Toeplitz
matrix Toe and a vector x. The complexity remains O(n^2), even though only the
first column of the matrix is stored.
"""
function matvec(Toe::bststb_1d, x::Vector{T})
  Toe_x = zeros(T, Toe.n)
  for i in 1:Toe.n
    Toe_x[i] += dot(Toe.e1[1:Toe.n-i+1], x[i:Toe.n])
    Toe_x[i] += dot(Toe.e1[i:-1:2], x[1:i-1])
  end
  return Toe_x
end

"""
get_cov_toeplitz_1d(n, cov)

Assembles a Toeplitz structure of a stationnary covariance cov(Δx) evaluated
at Δx = 0, 1 / (n - 1), ..., (n - 2) / (n - 1), 1.
"""
function get_cov_toeplitz_1d(n::Int, cov)
  dx = 1 / (n - 1)
  Te1 = Vector{T}(undef, n)
  for i in 1:n
    Te1[i] = cov((i - 1) * dx)
  end
  return bststb_1d(n, Te1)
end

import Base: *
(*)(Toe::bststb_1d, x::Vector{T}) = matvec(Toe::bststb_1d, x::Vector{T})

"""
bscscb_1d(n, m, hCe1, n_pos)

Structure for symmetric m-by-m circulant embedding of a symmetric n-by-n
Toeplitz matrix. Only the DFT hCe1 of the first column of the circulant matrix
is stored.
"""
struct bscscb_1d
  n::Int
  m::Int
  hCe1::Vector{T}
end

function get_pos_evals(C::bscscb_1d)
  nothing
end

"""
matvec(C, x)

Performs a matrix-vector product between a n-by-n symmetric Toeplitz matrix Toe
and a vector x, using a circulant embedding structure C of Toe.
"""
function matvec(C::bscscb_1d, x::Vector{T})
  x_0_padded = zeros(T, C.m)
  x_0_padded[1:C.n] = x
  y = C.hCe1 .* rfft(x_0_padded)
  q = irfft(y, C.m)
  return q[1:C.n]
end

"""
sample(C)

Draws a realization of the n-by-n symmetric Toeplitz matrix Toe, using a
circulant embedding structure C of Toe.
"""
function sample(C::bscscb_1d)
  x_0_padded = zeros(T, C.m)
  x_0_padded[1:C.n] = randn(T, C.n)
  y = sqrt.(C.hCe1) .* rfft(x_0_padded)
  q = irfft(y, C.m)
  return q[1:C.n]
end

"""
get_mce_1d(Toe)

Assembles the m-by-m minimally circulant embedding structure of a given symmetric
n-by-n Toeplitz structure (m = 2 * (n - 1)).
"""
function get_mce_1d(Toe::bststb_1d)
  m = 2 * (Toe.n - 1)
  #
  Ce1 = Vector{T}(undef, m)
  Ce1[1:Toe.n] = Toe.e1
  for j in 2:(Toe.n-1)
    Ce1[2 * (Toe.n - 1) - j + 2] = Ce1[j]
  end
  #
  hCe1 = real(rfft(Ce1))
  #
  trace_pos = 0.
  n_pos = Toe.n
  for j in 1:Toe.n
    al = hCe1[j]
    if al > 2 * eps(T)
      if j > 1
        trace_pos += 2 * al
      else
        trace_pos += al
      end
    else
      n_pos = j - 1
      break
    end
  end
  hCe1[n_pos+1:end] .= 0
  energy = trace_pos / (2 * (Toe.n - 1)), n_pos
  #
  return bscscb_1d(Toe.n, m, hCe1)
end

"""
get_mce_1d(n, cov)

Assembles the minimally circulant embedding structure for the symmetric
Toeplitz structure of a given stationary covariance cov(Δx). Faster and less
memory consuming than get_mce_1d(Toe).
"""
function get_mce_1d(n::Int, cov)

  dx = 1 / (n - 1)
  Te1 = Vector{T}(undef, n)
  for i in 1:n
    Te1[i] = cov((i - 1) * dx)
  end
  Toe = bststb_1d(n, Te1)


  m = 2 * (Toe.n - 1)
  Ce1 = Vector{T}(undef, m)
  Ce1[1:Toe.n] = Toe.e1
  for j in 2:(Toe.n-1)
    Ce1[2 * (Toe.n - 1) - j + 2] = Ce1[j]
  end
  hCe1 = real(rfft(Ce1))
  #
  trace_pos = 0.
  n_pos = Toe.n
  for j in 1:Toe.n
    al = hCe1[j]
    if al > 2 * eps(T)
      if j > 1
        trace_pos += 2 * al
      else
        trace_pos += al
      end
    else
      n_pos = j - 1
      break
    end
  end
  hCe1[n_pos+1:end] .= 0
  energy = trace_pos / (2 * (Toe.n - 1)), n_pos
  #
  return bscscb_1d(Toe.n, m, hCe1, n_pos)
end
