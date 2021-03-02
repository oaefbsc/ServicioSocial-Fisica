using CUDAdrv, CUDAnative, CuArrays, Test, BenchmarkTools

function vadd2(a, b, c)
    indice = (blockIdx().x-1) * blockDim().x + threadIdx().x
    paso = blockDim().x * gridDim().x
    for i = indice:paso:length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

function bench_cuda(d_a, d_b, d_c, threads, blocks)
        @cuda blocks = blocks threads = threads vadd2(d_a, d_b, d_c)
        return Array(d_c)
end

dims = (3000, 4000)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)

# How many threads we are using, one for each vector element
len = prod(dims)
bloques = ceil(Int64, len/1024)
hilos = ceil(Int64, len/bloques)
# Using cuda!
@btime bench_cuda(d_a, d_b, d_c, hilos, bloques) # ----> ¿Por qué a veces se usa '$' en los argumentos?
@btime d_a + d_b
@btime a + b

