# Library to see if the programm is working well
using Test
# GPU libraries
using CUDAdrv, CUDAnative, CuArrays
#const CuArray = CUDAnative.CuHostArray  # real applications: use CuArrays.jl

function vadd2(a, b, c)
    # blockDim().x <- Threads per block
    # threadIdx().x <- Thread id
    # blockIdx().x-1 <- Block id - 1, i think we beggin in 1 an we want to 
    # start in 0.
    indice = (blockIdx().x-1) * blockDim().x + threadIdx().x
    paso = blockDim().x * gridDim().x
    for i = indice:paso:length(a)
        c[i] = a[i] + b[i]
    end
    return
end

dims = (33,33)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)

# How many threads we are using, one for each vector element
len = prod(dims)
blocks = ceil(Int64, len/1024)
threads = ceil(Int64, len/blocks)
# Using cuda!
@cuda blocks = blocks threads = threads  vadd2(d_a, d_b, d_c)
synchronize()
c = Array(d_c)
@test c ≈ a+b


