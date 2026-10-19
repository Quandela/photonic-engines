"""TimeLoopLayer: the MerLin-style trainable layer on the native time-loop engine."""
import torch

from photonic_engines.adapters.merlin import TimeLoopLayer
from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit


def test_readouts_shapes_and_gradients():
    c = TimeLoopCircuit(5, (1, 2))
    occ = [1, 1, 0, 0, 0]
    x = torch.tensor([[0.3, 1.2], [2.0, 0.1], [1.0, 1.0]], dtype=torch.float64)
    for readout, size in [("clicks", 5), ("mean", 5), ("marginals", 15), ("distribution", 15)]:
        layer = TimeLoopLayer(c, occ, input_size=2, readout=readout,
                              loss=None if readout == "distribution" else {"loop": [0.9, 0.95], "detector": 0.8})
        y = layer(x)
        assert y.shape == (3, size) and layer.output_size == size
        y.sum().backward()
        assert layer.theta.grad is not None and float(layer.theta.grad.abs().sum()) > 0
        if readout == "distribution":
            assert torch.allclose(y.sum(1), torch.ones(3, dtype=y.dtype))
    # the encoded row equals the kernel evaluated at theta + x on the encode slots (flushed)
    layer = TimeLoopLayer(c, occ, input_size=2, readout="clicks", encode_slots=[(0, 0), (1, 1)])
    theta = layer.theta.detach().clone()
    theta[0, 0] += x[0, 0]
    theta[1, 1] += x[0, 1]
    theta, phi = c.flush_settings(theta, layer.phi.detach())
    ref = 1 - LoopRegisterKernel(c, occ).bin_marginals(theta, phi)[:, 0]
    assert torch.allclose(layer(x[:1])[0], ref, atol=1e-12)
    assert layer.sample(7, x[0], threshold=True).shape == (7, 5)


def test_fits_hidden_settings():
    """Train theta/phi so that the click pattern of 4 inputs matches a hidden device."""
    c = TimeLoopCircuit(5, (1,))
    occ = [1, 1, 0, 0, 0]
    x = torch.tensor([[0.2], [0.9], [1.7], [2.6]], dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    target_layer = TimeLoopLayer(c, occ, input_size=1, theta=torch.rand(1, 5, generator=g, dtype=torch.float64) * 3)
    with torch.no_grad():
        target = target_layer(x)
    layer = TimeLoopLayer(c, occ, input_size=1, theta=torch.full((1, 5), 0.7, dtype=torch.float64))
    opt = torch.optim.Adam(layer.parameters(), lr=0.1)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
