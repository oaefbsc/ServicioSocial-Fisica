using CUDAdrv, CUDAnative, CuArrays, Test

function prod_escalar(escalar, cu_array_in, cu_array_out)
        inicio = (blockIdx().x-1) * blockDim().x + threadIdx().x
        paso = blockDim().x * gridDim().x
        for i = inicio:paso:length(cu_array_in)
                @inbounds cu_array_out[i] = escalar*cu_array_in[i]
        end
        return
end

dims = (33, 33)
a = round.(rand(Float32, dims) * 100)
b = similar(a)
c = 3.0f0

d_a = CuArray(a)
d_b = CuArray(b)

len = prod(dims)
bloques = ceil(Int64, len/1024)
@cuda blocks = bloques threads = ceil(Int64, len/bloques) prod_escalar(c, d_a, d_b)
b = Array(d_b)
@test b ≈ c*a

