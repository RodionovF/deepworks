#pragma once

#include "kernels.hpp"

namespace deepworks {

/*
 * Base class for all optimizers
 */
class Optimizer {
public:
    virtual void CPUstep(Matrix& W, const ConstMatrix& dW) = 0;
};

/*
 * Realization of simple stochastic gradient descent
 */
class SGD : public Optimizer {
public:
    explicit SGD(float lr);

    void CPUstep(Matrix& W, const ConstMatrix& dW) override;
    float get_learning_rate() const;
    void set_learning_rate(float lr);

private:
    float learning_rate;
};

} // namespace deepworks