#We would like to thanks the financial support of PAPIIT-IA104720

using Pkg; Pkg.activate("/homen1/oscar_aam/.julia/environments/CNN")

using Images, ImageFeatures # We have to Pkg.add("Netpbm") because our images are in pgm format
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: partition
using Printf, BSON
using BenchmarkTools

arc = 64

train_imgs = []
train_labels = []
path_train = "/almac/oscar_aam/Yale/CroppedYale$(arc)/train/"
for directory = readdir(path_train)
    for file = readdir(path_train*directory)
        #println(path_train*directory*"/"*file, directory[end-1:end])
        push!(train_imgs, Float64.(load(path_train*directory*"/"*file)))
        label = Meta.parse(directory[end-1:end])
        if label < 14
            push!(train_labels, label)
        else
            push!(train_labels, label-1)
        end
    end
end

test_imgs = []
test_labels = []
path_train = "/almac/oscar_aam/Yale/CroppedYale$(arc)/test/"
for directory = readdir(path_train)
    for file = readdir(path_train*directory)
        push!(test_imgs, Float64.(load(path_train*directory*"/"*file)))
        label = Meta.parse(directory[end-1:end])
        if label < 14
            push!(test_labels, label)
        else
            push!(test_labels, label-1)
        end
    end
end

# Model definition
# See: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
imgsize = (arc, arc, 1)
cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32]))
model = Chain(
    # First convolution, operating upon a 192×192image
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)), #maxpooling

    # Second convolution, operating upon a 96×96 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)), #maxpooling

    # Third convolution, operating upon a 48×48 image
    Conv((3, 3), 32=>32,pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (24, 24, 32, N)
    # which is where we get the 18432 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(prod(cnn_output_size), 38),

    # Softmax to get probabilities
    softmax,
)

# Load on gpu (if available)
#model = gpu(model);

# Batching 
# See: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs, labels)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], labels)
    return (X_batch, Y_batch)
end
# The CNN only "sees" 128 images at each training cycle:
batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
# train set in the form of batches
train_set = [make_minibatch(train_imgs, train_labels, i, 1:38) for i in mb_idxs];
# train set in one-go: used to calculate accuracy with the train set
train_set_full = make_minibatch(train_imgs, train_labels, 1:length(train_imgs), 1:38);
# test set: to check we do not overfit the train data:
test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs), 1:38);

# Loss function
# See: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
# `loss()` calculates the crossentropy loss between our prediction `y_hat`
function loss(x, y)
    # Add some noise to the image
    # we reduce the risk of overfitting the train sample by doing so:
    x_aug = x .+ 0.1f0*randn(eltype(x), size(x))

    y_hat = model(x_aug)
    return crossentropy(y_hat, y)
end
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# ADAM optimizer
opt = ADAM(0.001);

# Training loop
# See: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
best_acc = 0.0
last_improvement = 0
accuracy_target = 0.95  #Set an accuracy target. When reached, we stop training.
max_epochs = 200 #Maximum
for epoch_idx in 1:max_epochs
    global best_acc, last_improvement
    # Train for a single epoch
    println("Epoch $(epoch_idx)")
    @btime Flux.train!(loss, Flux.params(model), train_set, opt) 

    # Calculate accuracy:
    acc = accuracy(train_set_full...)
    @info(@sprintf("[%d]: Train accuracy: %.4f", epoch_idx, acc))
    
    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= accuracy_target
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
pred = model(test_set[1]); 

# Function to get the row index of the max value: 
f1(x) = getindex.(argmax(x, dims=1), 1) # Final predicted value is the one with the maximum probability: 
pred = f1(pred); #minus 1, because the first digit is 0 (not 1)

println("Predicted value = $(pred[1])")
