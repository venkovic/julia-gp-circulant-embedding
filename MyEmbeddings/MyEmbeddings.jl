module MyEmbeddings

export get_cov_toeplitz_1d, get_mce_1d
#export get_cov_toeplitz_2d, get_mce_2d
export matvec#, sample

const T = Float64
const T32 = Float32
const T16 = Float16

include("1DEmbeddings.jl")
#include("2DEmbeddings.jl")

end
