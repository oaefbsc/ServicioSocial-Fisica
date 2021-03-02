using CUDAdrv, CuArrays, CUDAnative
using Test

function kernel(C, A, B)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y

        if i <= size(A,1) && j <= size(B,2)
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Ctmp = convert(promote_type(R, typeof(z2)), z2)
            for k in 1:size(A,2)
                Ctmp += A[i, k]*B[k, j]
            end
            C[i,j] = Ctmp
        end

        return
end

dim_x = 4
dims = (3, dim_x)
A = round.(rand(Float32, dims) * 100)
B = round.(rand(Float32, dim_x, size(A,1)) * 100)
C = Array{Float32}(undef, size(A,1), size(B,2))

d_A = CuArray(A)
d_B = CuArray(B)
d_C = CuArray(C)

@show A
@show B

max_threads = 256
threads_x = min(max_threads, size(C, 1))
threads_y = min(max_threads ÷ threads_x, size(C, 2))
threads = (threads_x, threads_y)
blocks = ceil.(Int, (size(C, 1), size(C, 2)) ./ threads)

@cuda threads=threads blocks=blocks kernel(d_C, d_A, d_B)
C = Array(d_C)
@test C ≈ A*B

