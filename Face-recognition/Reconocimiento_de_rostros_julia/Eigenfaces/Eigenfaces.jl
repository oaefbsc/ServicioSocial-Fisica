#We would like to thanks the financial support of PAPIIT-IA104720

using Pkg
#Pkg.activate("/home/oscar/.julia/environments/Eigenfaces")
Pkg.activate("C:/Users/Óscar Alvarado/.julia/environments/Eigenfaces") # To use the correct environment

using Images, ImageFeatures # We have to Pkg.add("Netpbm") because our images are in pgm format
using Statistics
using LinearAlgebra
using BenchmarkTools
#using Plots
#gr()

dim = 192

train_imgs = zeros(2339, dim, dim)
train_labels = []
path_train = "../datasets/Yale/CroppedYale$(dim)/train/"
count1 = 1
for directory = readdir(path_train)
    for file = readdir(path_train*directory)
        #println(path_train*directory*"/"*file, directory[end-1:end])
        train_imgs[count1,:,:] =  load(path_train*directory*"/"*file)
        push!(train_labels, Meta.parse(directory[end-1:end]))
        count1 += 1
    end
end

test_imgs = zeros(76, dim, dim)
test_labels = []
path_test = "../datasets/Yale/CroppedYale$(dim)/test/"
count2 = 1
for directory = readdir(path_test)
    for file = readdir(path_test*directory)
        test_imgs[count2, :, :] = load(path_test*directory*"/"*file)
        push!(test_labels, Meta.parse(directory[end-1:end]))
        count2 += 1
    end
end

Gray.(test_imgs[1,:,:]) # We have to Pkg.add("ImageMagick") to see the image

# This is the mean face
mean_tensor = mean(train_imgs, dims = 1)[1,:,:]; # We get a 3D array (1x192x192) but we just need the 192x192
Gray.(mean_tensor)

tensor_PCA = zeros(2339, dim, dim)
[tensor_PCA[idx,:,:]= train_imgs[idx,:,:] .- mean_tensor for idx in 1:2339];

Gray.(tensor_PCA[1,:,:])

function T2M(tensor)
    dims = size(tensor)
    return reshape(tensor, (dims[1], dims[2]*dims[3]))
end

function M2T(matrix)
    dims = size(matrix)
    return reshape(matrix, (dims[1], trunc(Int64, sqrt(dims[2])), trunc(Int64, sqrt(dims[2]))))
end

X = T2M(tensor_PCA);

U, Sigma, VT = svd(X)
eigenfaces = M2T(VT'); # The eigenfaces as a tensor

n_eigen = ceil(Int64, size(VT)[2]/70)
proyection =  T2M(train_imgs) * VT[:, 1:n_eigen];

acc = []
for n_eigen in 1:size(VT)[2]
    i = 0
    proyection =  T2M(train_imgs) * VT[:, 1:n_eigen];
    for idx in 1:size(test_imgs)[1]
        test = reshape(test_imgs[idx,:,:], (1, dim^2)) * VT[:, 1:n_eigen]
        dists = sum((test .- proyection) .^ 2, dims = 2)
        pred = train_labels[argmin(dists)]
        #@show pred, test_labels[idx]
        if pred == test_labels[idx]
            i += 1
        end
    end
    @show i/76*100, n_eigen
    push!(acc, i/76*100)
    if i/76*100 == 100.0
        print(n_eigen)
        break
    end
end
