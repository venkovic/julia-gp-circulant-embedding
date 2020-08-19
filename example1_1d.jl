push!(LOAD_PATH, "./MyEmbeddings")
push!(LOAD_PATH, "./MyCorrelations")

using Random: seed!
using MyCorrelations: exp_1d
using MyEmbeddings: get_cov_toeplitz_1d, get_mce_1d

T = Float64

seed!(1234)

const n = 10_000
const L = .01

function cov(Δx::T)
  exp_1d(Δx, L)
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
