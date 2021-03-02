#using Pkg; Pkg.activate("/homen1/oscar_aam/.julia/environments/CNN")
using Pkg; Pkg.activate("/homen1/oscar_aam/.julia/environments/v1.4")

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: partition
using Printf, BSON
using BenchmarkTools

imgs = MNIST.images(:train);
labels = onehotbatch(MNIST.labels(:train), 0:9);

# Partition into batches of size 1,000
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
         for i in partition(1:60_000, 128)];

train = gpu.(train);

# Prepare test set (first 1,000 images)
tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu;
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu;

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
    Dense(288, 10), softmax) |> gpu

# Loss function
# See: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
# `loss()` calculates the crossentropy loss between our prediction `y_hat`
loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)

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
    @btime Flux.train!(loss, Flux.params(model), train, opt) 

    # Calculate accuracy:
    #acc = accuracy(train_set_full...)
    #@info(@sprintf("[%d]: Train accuracy: %.4f", epoch_idx, acc))

    # Calculate accuracy:
    acc = accuracy(tX, tY)
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
pred = model(tX); 

# Function to get the row index of the max value: 
f1(x) = getindex.(argmax(x, dims=1), 1) # Final predicted value is the one with the maximum probability: 
pred = f1(pred) .- 1; #minus 1, because the first digit is 0 (not 1)

println("Predicted value = $(pred[1])")