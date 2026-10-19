py::class_<BC_MixedExpertsBsz1, std::shared_ptr<BC_MixedExpertsBsz1>>(m, "BC_MixedExpertsBsz1").def
(
    py::init<
        std::vector<std::shared_ptr<BC_LinearEXL3>>,
        std::vector<std::shared_ptr<BC_LinearEXL3>>,
        std::vector<std::shared_ptr<BC_LinearEXL3>>,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        int,
        float
    >(),
    py::arg("gates"),
    py::arg("ups"),
    py::arg("downs"),
    py::arg("interm_g"),
    py::arg("interm_u"),
    py::arg("interm_a"),
    py::arg("out_d"),
    py::arg("act"),
    py::arg("act_limit")
)
.def("run", &BC_MixedExpertsBsz1::run);
