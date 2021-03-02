using CUDAdrv, CUDAnative, CuArrays, Test, BenchmarkTools

function prod_vect(cu_vector, cu_array_in, cu_array_out)
        inicio = (blockIdx().x-1) * blockDim().x + threadIdx().x
        paso = blockDim().x * gridDim().x
        renglones = size(cu_array_in)[1]
        for i = inicio:paso:length(cu_array_in)
                @inbounds cu_array_out[i] = cu_array_in[i] * cu_vector[Int64(ceil(i/renglones))]
        end
        return
end

function bench_cuda(d_x, d_A, d_b, threads, blocks)
        @cuda blocks = blocks threads = threads prod_vect(d_x, d_A, d_b)
        return sum(Array(d_b), dims = 2)
end

dim_x = 4000
dims = (3000, dim_x)
A = round.(rand(Float32, dims) * 100)
x = round.(rand(Float32, dim_x) * 100)
b = similar(A)

d_A = CuArray(A)
d_x = CuArray(x)
d_b = CuArray(b)

len = prod(dims)
bloques = ceil(Int64, len/1024) 
hilos = ceil(Int64, len/bloques)
@btime bench_cuda(d_x, d_A, d_b, hilos, bloques)
@btime d_A*d_x
@btime A*x



