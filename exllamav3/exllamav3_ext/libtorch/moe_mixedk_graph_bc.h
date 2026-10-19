py::class_<BC_MixedKExperts, std::shared_ptr<BC_MixedKExperts>>(m, "BC_MixedKExperts").def
(
    py::init<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        std::vector<at::Tensor>,
        at::Tensor,
        std::shared_ptr<BC_GatedMLP>,
        c10::optional<at::Tensor>,
        int,
        int,
        float
    >(),
    py::arg("remap"),
    py::arg("yh"),
    py::arg("interm_g"),
    py::arg("interm_u"),
    py::arg("interm_a"),
    py::arg("out_d"),
    py::arg("y_static"),
    py::arg("shared_experts"),
    py::arg("out_sh"),
    py::arg("top_k"),
    py::arg("act"),
    py::arg("act_limit")
)
.def("add_group", &BC_MixedKExperts::add_group)
.def("run", &BC_MixedKExperts::run);
