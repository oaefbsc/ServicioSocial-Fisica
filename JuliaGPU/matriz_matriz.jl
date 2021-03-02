using CUDAdrv, CUDAnative, CuArrays, Test, BenchmarkTools

function prod_mat(cu_A, cu_B_in, cu_C)
        inicio = (blockIdx().x-1) * blockDim().x + threadIdx().x
        paso = blockDim().x * gridDim().x
        renglones = size(cu_array_in)[1]
	columnas = size(cu_array_in)[2]
        for i = inicio:paso:length(cu_array_in)
                @inbounds cu_C[i] = cu_A[i] * cu_B[i]
        end
        return
end

function bench_cuda(d_x, d_A, d_b, threads, blocks)
        @cuda blocks = blocks threads = threads prod_mat(d_A, d_B, d_C)
        return sum(Array(d_b), dims = 2)
end

N = 32
M = 32
K = 3
dims1 = (M, N)
dims2 = (N, K)
A = round.(rand(Float32, dims1) * 100)
B = round.(rand(Float32, dims2) * 100)
C = Array{Union{Missing, Array{Int64,1}}}(missing, M, K)

d_A = CuArray(A)
d_x = CuArray(x)
d_b = CuArray(b)

len = prod(dims)
bloques = ceil(Int64, len/1024) 
hilos = ceil(Int64, dims/bloques)
@btime bench_cuda(d_x, d_A, d_b, hilos, bloques)
@btime A*x



