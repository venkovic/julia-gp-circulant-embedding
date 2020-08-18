# 1D square exponential kernels
function sexp_1d(Δx::T, L::T)
  return exp(- Δx ^ 2 / L ^ 2)
end
function sexp_1d(Δx::Vector{T}, L::T)
  return exp.(- Δx .^ 2 / L ^ 2)
end
function sexp_1d(x::T, y::T, L::T)
  return exp(- (x  - y) ^ 2 / L ^ 2)
end
function sexp_1d(x::Vector{T}, y::Vector{T}, L::T)
  return exp.(- (x - y) .^ 2 / L ^ 2)
end

# 1D exponential kernels
function exp_1d(Δx::T, L::T)
  return exp(- abs(Δx) / L)
end
function exp_1d(Δx::Vector{T}, L::T)
  return exp.(- abs.(Δx) / L)
end
function exp_1d(x::T, y::T, L::T)
  return exp(- abs(x - y) / L)
end
function exp_1d(x::Vector{T}, y::Vector{T}, L::T)
  return exp.(- abs.(x - y) / L)
end
