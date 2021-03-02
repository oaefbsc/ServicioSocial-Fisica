#using CUDAdrv, CUDAnative, CuArrays, Test, BenchmarkTools
using CUDA, BenchmarkTools, Test

function prod_escalar(escalar, cu_array_in, cu_array_out)
        inicio = (blockIdx().x-1) * blockDim().x + threadIdx().x
        paso = blockDim().x * gridDim().x
        for i = inicio:paso:length(cu_array_in)
                @inbounds cu_array_out[i] = escalar*cu_array_in[i]
        end
        return
end

function bench_cuda(d_a, d_b, c, threads, blocks)
        @cuda blocks = bloques threads = threads prod_escalar(c, d_a, d_b)
        return Array(d_a)
end

dims = (3000, 4000)
a = round.(rand(Float32, dims) * 100)
b = similar(a)
c = 3.0f0

d_a = CuArray(a)
d_b = CuArray(b)

len = prod(dims)
bloques = ceil(Int64, len/1024)
hilos = ceil(Int64, len/bloques)
@btime bench_cuda(d_a, d_b, c, hilos, bloques)
@btime c*d_a
@btime c*a



