#pragma once

#include "runtime/cpu/kernels/kernels.hpp"

namespace deepworks {

/*
 * Base class for all loss functions
 */
class Loss {
public:
    virtual void CPUForward(const ConstMatrix& X, const ConstVector& target, Matrix& loss) = 0;
    virtual void CPUBackward(const ConstMatrix& X, const ConstVector& target, Matrix& grad_output) = 0;
};

/*
 * Realization of CrossEntropyLoss
 * This criterion combines LogSoftmax and NLLLoss in one single class
 */
class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss() = default;
    
    void CPUForward(const ConstMatrix& X, const ConstVector& target, Matrix& loss);
    void CPUBackward(const ConstMatrix& X, const ConstVector& target, Matrix& grad_output);
};

} // namespace deepworks