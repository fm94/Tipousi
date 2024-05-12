#include "loss/mse.hpp"

#include "activation/relu.hpp"
#include "activation/softmax.hpp"

#include "layer/dense.hpp"
//#include "layer/input.hpp"

#include "model/sequential.hpp"

#include <memory>

using namespace Tipousi;

int main(int argc, char const *argv[])
{
    // tests

    // loss functions
    Loss::MSE mse;

    // activation functions
    Activation::ReLU relu;
    Activation::Softmax softmax;

    // layers
    Layer::Dense dense(16, 32); // 16 inputs, 32 outputs

    // models
    std::vector<Op*> model {};
    Model::Sequential sequential(model);

    // build a model - current idea
    // TODO: unique ptrs, handle ownership?
    // use raw pointers and ownership should go to the next layer
    // when the next layer is destroyed it destroys the previous
    // layer and so on until the first one
    // so to destroy the model we need to destroy the sequential which then 
    // triggers this chain reaction and we get no memory leaks

    //auto input = std::make_unique<Input>(5);
    // auto x1 = std::make_unique<Layer::Dense>(5, 32)(input);
    // auto x2 = std::make_unique<Activation::ReLU>()(x1);
    // auto x3 = std::make_unique<Layer::Dense>(32, 2)(x2);
    // auto output = std::make_unique<Activation::Softmax>()(x3);
    // Model::Sequential model(input, output);
}
