#We would like to thanks the financial support of PAPIIT-IA104720
#using Pkg; Pkg.activate("/homen1/oscar_aam/.julia/environments/CNN")
using Pkg; Pkg.activate("/homen1/oscar_aam/.julia/environments/v1.4")

#using Images, ImageFeatures # We have to Pkg.add("Netpbm") because our images are in pgm format
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: partition
using Printf, BSON
using BenchmarkTools

arc = 28

min_max_scale(a) = (a .- min(a...))/(max(a...)-min(a...)) # The Gray() function only works with 0-1 arrays

function raw2array(path, arc)
    io = open(path, "r")
    raw_data = read(io)
    close(io)
    len = length(raw_data)
    raiz = trunc(Int64, sqrt(len))
    if raiz == arc # If the image is squared
        return min_max_scale((convert.(Float64, reshape(raw_data, raiz, raiz)))')
    else
        return zeros(arc, arc)
    end
end


imgs1 = []
train_labels1 = []
path_train = "/almac/oscar_aam/Yale/CroppedYale28raw/train/"
for directory = readdir(path_train)
    for file = readdir(path_train*directory)
        label = Meta.parse(directory[end-1:end])
        if label <= 10
            push!(imgs1, raw2array(path_train*directory*"/"*file, arc))
            push!(train_labels1, label - 1)
        end
    end
end
train_labels1 = Int64.(train_labels1);
labels1 = onehotbatch(train_labels1, 0:9);
# Partition into batches of size 128
train1 = [(cat(imgs1[i]..., dims = 4), labels1[:,i])
         for i in partition(1:length(train_labels1), 64)];

train1 = gpu.(train1);

test_imgs1 = []
test_labels1 = []
path_train = "/almac/oscar_aam/Yale/CroppedYale28raw/test/"
for directory = readdir(path_train)
    for file = readdir(path_train*directory)
        label = Meta.parse(directory[end-1:end])
        if label <= 10
            push!(test_imgs1, raw2array(path_train*directory*"/"*file, arc))
            push!(test_labels1, label - 1)
        end
    end
end
test_labels1 = Int64.(test_labels1);
# Prepare test set
tX1 = cat(test_imgs1..., dims = 4) |> gpu;
tY1 = onehotbatch(test_labels1, 0:9) |> gpu;

imgsize = (arc, arc, 1)
cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32]))
model1 = Chain(
    # First convolution, operating upon a 28×28image
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)), #maxpooling

    # Second convolution, operating upon a 14×14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)), #maxpooling

    # Third convolution, operating upon a 7×7 image
    Conv((3, 3), 32=>32,pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 18432 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax) |> gpu

# Loss function
# See: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
# `loss()` calculates the crossentropy loss between our prediction `y_hat`
loss(x, y) = crossentropy(model1(x), y)
accuracy(x, y) = mean(onecold(model1(x)) .== onecold(y))
evalcb = throttle(() -> @show(accuracy(tX1, tY1)), 10)

# ADAM optimizer
opt = ADAM(0.001);

# Training loop
# See: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
best_acc = 0.0
last_improvement = 0
accuracy_target = 0.88  #Set an accuracy target. When reached, we stop training.
max_epochs = 200 #Maximum
for epoch_idx in 1:max_epochs
    global best_acc, last_improvement
    # Train for a single epoch
    println("Epoch $(epoch_idx)")
    @btime Flux.train!(loss, Flux.params(model1), train1, opt)

    # Calculate accuracy:
    #acc = accuracy(train_set_full...)
    #@info(@sprintf("[%d]: Train accuracy: %.4f", epoch_idx, acc))

    # Calculate accuracy:
    acc1 = accuracy(tX1, tY1)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc1))

    # If our accuracy is good enough, quit out.
    if acc1 >= accuracy_target
        @info(" -> Early-exiting: We reached our target accuracy of $(accuracy_target*100)%")
        @show epoch_idx
        break
    end

    if epoch_idx - last_improvement >= max_epochs
        @warn(" -> We're calling this converged.")
        break
    end
end

# Get predictions and convert data to Array:
pred1 = model1(tX1); 

# Function to get the row index of the max value:
f1(x) = getindex.(argmax(x, dims=1), 1) # Final predicted value is the one with the maximum probability:
pred1 = f1(pred1) .- 1; 

println("Predicted value = $(pred1[1])")

