module MyCovariances

export sexp_1d, exp_1d, mattern_1d
#export sexp_2d, exp_2d, mattern_2d

const T = Float64
const T32 = Float32
const T16 = Float16

include("1DCorrelations.jl")
#include("2DCorrelations.jl")

end
