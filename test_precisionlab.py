import unittest
import torch
from precisionlab import TinyDecoder, fp8_qdq, samples, evaluate


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_zero(self):
        torch.testing.assert_close(fp8_qdq(torch.zeros(4)), torch.zeros(4))

    def test_nonfinite(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                fp8_qdq(torch.tensor([value]))

    def test_empty(self):
        with self.assertRaises(ValueError):
            fp8_qdq(torch.empty(0))

    def test_extreme_scale(self):
        self.assertTrue(torch.isfinite(fp8_qdq(torch.tensor([-1e20, 0., 1e20]))).all())

    def test_error(self):
        x = torch.linspace(-10, 10, 100)
        self.assertLess((fp8_qdq(x) - x).norm() / x.norm(), 0.05)

    def test_straight_through_gradient(self):
        x = torch.randn(8, requires_grad=True)
        fp8_qdq(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones(8))

    def test_causality(self):
        model = TinyDecoder().eval()
        x, _ = samples(2, 1)
        altered = x.clone()
        altered[:, 8:] = (altered[:, 8:] + 1) % 32
        torch.testing.assert_close(model(x)[:, :8], model(altered)[:, :8])

    def test_shape(self):
        self.assertEqual(tuple(TinyDecoder()(samples(2, 1)[0]).shape), (2, 16, 32))

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            TinyDecoder()(torch.zeros(2, 17, dtype=torch.long))

    def test_finite_emulated_backward(self):
        model = TinyDecoder().precision(True)
        model(samples(2, 1)[0]).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_seeded_samples(self):
        x, y = samples(2, 1)
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        torch.testing.assert_close(x, samples(2, 1)[0])

    def test_evaluate(self):
        self.assertGreater(evaluate(TinyDecoder(), samples(2, 1))['perplexity'], 0)


if __name__ == '__main__':
    unittest.main()
