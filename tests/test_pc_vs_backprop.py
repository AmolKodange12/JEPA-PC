import unittest
import torch
import torch.nn as nn
import torch.optim as optim

class TestPCVsBackprop(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Synthetic MNIST batch input
        self.x_mock = torch.rand(self.batch_size, 784).to(self.device)
        self.y_mock = torch.randint(0, 10, (self.batch_size,)).to(self.device)
        self.y_onehot = F.one_hot(self.y_mock, num_classes=10).float().to(self.device)

    def test_pc_shapes_and_contract_compliance(self):
        """Verifies that the custom PC layers match contract specifications."""
        pc_net = MNISTPCNetwork(use_tanh=True).to(self.device)
        
        # Step 1: Initialize states via forward pass
        pc_net.init_states(self.x_mock)
        self.assertEqual(pc_net.layers[0].r.shape, (self.batch_size, 256))
        self.assertEqual(pc_net.layers[1].r.shape, (self.batch_size, 64))
        self.assertEqual(pc_net.layers[2].r.shape, (self.batch_size, 10))
        
        # Step 2: Ensure inference steps execute without altering parameter shapes
        energies = pc_net.infer(self.x_mock, n_steps=5, lr_infer=0.1)
        self.assertEqual(len(energies), 5)
        # Energy must decrease or stay stable if alignment is correct
        self.assertTrue(energies[-1] <= energies[0])

    def test_comparative_optimization_step(self):
        """Compares single-step weight evolution behavior between PC and Backprop."""
        pc_net = MNISTPCNetwork(use_tanh=False).to(self.device)
        bp_net = BackpropNetwork(use_tanh=False).to(self.device)
        
        # Test backprop step
        optimizer = optim.SGD(bp_net.parameters(), lr=0.01)
        out_bp = bp_net(self.x_mock)
        loss = F.mse_loss(out_bp, self.y_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Test Predictive Coding convergence and weight update step
        pc_net.init_states(self.x_mock)
        # Clamp top layer to label during training
        pc_net.layers[-1].r = self.y_onehot.clone()
        pc_net.infer(self.x_mock, n_steps=20, lr_infer=0.1)
        
        # Update weights locally using errors
        w_before = pc_net.layers[0].W.clone()
        pc_net.update_all_weights(self.x_mock, lr_weights=0.01)
        w_after = pc_net.layers[0].W
        
        # Assert weights actually modified
        self.assertFalse(torch.equal(w_before, w_after))

if __name__ == "__main__":
    unittest.main()