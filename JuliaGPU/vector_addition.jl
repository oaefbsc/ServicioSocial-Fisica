# Library to see if the programm is working well
using Test
# GPU libraries
using CUDAdrv, CUDAnative, CuArrays
#const CuArray = CUDAnative.CuHostArray  # real applications: use CuArrays.jl

function vadd(a, b, c)
    # blockDim().x <- Threads per block
    # threadIdx().x <- Thread id
    # blockIdx().x-1 <- Block id - 1, i think we beggin in 1 an we want to 
    # start in 0.
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @cuprintf("block %ld, thread %ld, blockdim %ld!\n", Int64(blockIdx().x-1), Int64(threadIdx().x), Int64(blockDim().x))
    c[i] = a[i] + b[i]
    return
end

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)
@show a
@show b

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)

# How many threads we are using, one for each vector element
len = prod(dims)
# Using cuda!
@cuda threads = len vadd(d_a, d_b, d_c)
c = Array(d_c)
@test a+b ≈ c


