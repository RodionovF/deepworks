#include "loss.hpp"

namespace deepworks {

void CrossEntropyLoss::CPUForward(const ConstMatrix& X, const ConstVector& target, Matrix& loss) {
    int rows = X.rows();
    int cols = X.cols();

    Eigen::MatrixXf p = Eigen::MatrixXf::Zero(rows, cols);
    Matrix P(p.data(), rows, cols);

    CPUSoftmaxForward(X, P);

    // NB: use Eigen:seq
    for (int i = 0; i < cols; ++i) {
        Matrix(0, 0) -= std::log(P(i, target(0, i)));
    }
}

} // namespace deepworks