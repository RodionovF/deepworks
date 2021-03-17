#include <deepworks/tensor.hpp>

#include "model/graph.hpp"
#include "runtime/backend.hpp"
#include "runtime/cpu/layers/cpulayer.hpp"

#include <ade/graph.hpp>

namespace deepworks {
namespace cpu {

class CPUBackend : public IBackend {
public:
    CPUBackend(deepworks::graph::Graph& g);

    void forward (const std::vector<deepworks::Tensor>& inputs,
                        std::vector<deepworks::Tensor>& outputs) override;

    void backward(const std::vector<deepworks::Tensor>& inputs,
                        std::vector<deepworks::Tensor>& outputs) override;

private:
    void bind(const std::vector<deepworks::Tensor>& tensors,
              const std::vector<ade::NodeHandle>  & handles);

    deepworks::graph::Graph&      m_graph;
    deepworks::graph::TypedGraph  m_tgraph;

    std::vector<deepworks::Tensor> m_mem;
    std::vector<ade::NodeHandle>   m_ops;
};

} // namespace cpu
} // namespace deepworks
