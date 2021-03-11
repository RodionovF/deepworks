#pragma once

#include "kernels.hpp"

namespace deepworks {

/*
 * Base class for all loss functions
 */
    class Loss {
    public:
        virtual void CPUForward(const ConstMatrix& X, const ConstVector& target) = 0;
        virtual void CPUBackward(const ConstMatrix& X, const ConstVector& target) = 0;
    };

/*
 * Realization of CrossEntropyLoss
 * This criterion combines LogSoftmax and NLLLoss in one single class
 */
    class CrossEntropyLoss : public Loss {
    public:
        CrossEntropyLoss() = default;
    private:
        float learning_rate;
    };

} // namespace deepworks