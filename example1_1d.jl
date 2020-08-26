push!(LOAD_PATH, "./MyEmbeddings")
push!(LOAD_PATH, "./MyCorrelations")

using Random: seed!
using LinearAlgebra: norm2
using MyCorrelations: exp_1d, sexp_1d
using MyEmbeddings: get_cov_toeplitz_1d, get_mce_1d, matvec, sample
using NPZ: npzwrite

T = Float64

seed!(1234)

const n = 1_000
const L = .05

function cov(Δx::T)
  exp_1d(Δx, L)
  #sexp_1d(Δx, L)
end

K = get_cov_toeplitz_1d(n, cov)
q = rand(T, n)
Kq = K * q

mce_of_K = get_mce_1d(K)
Kq2 = matvec(mce_of_K, q)

println("\n")
println("Regulat matvec:")
println(Kq[1], ", ", Kq[2], ", ", Kq[3], ", ...")
println()
println("FFT-based matvec:")
println(Kq2[1], ", ", Kq2[2], ", ", Kq2[3], ", ...")
Kq .-= Kq2
println()
println("Error norm:")
println(norm2(Kq))


for k in 1:20
  q = sample(mce_of_K)
  npzwrite("data/real$(k).npy", q)
end


npzwrite("data/evals.npy", mce_of_K.hCe1)


# Doc
# Do fast version of get_mce_1d(n::Int, cov)
# Cases that need padding // Padding, only for SQRT?
#
#
# Simulate
# import into iterative mle
