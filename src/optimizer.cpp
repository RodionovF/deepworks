#include "optimizer.hpp"

namespace deepworks {

SGD::SGD(float lr) : learning_rate(lr) {}

void SGD::CPUstep(Matrix &W, const ConstMatrix &dW) {
    W -= learning_rate * dW;
}

float SGD::get_learning_rate() const {
    return learning_rate;
}

void SGD::set_learning_rate(float lr) {
    learning_rate = lr;
}

} // namespace deepworks
