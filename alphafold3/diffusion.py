import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GeneticDiffusionModuleBlock(nn.Module):
    """
    Diffusion Module from AlphaFold 3.

    This module directly predicts raw atom coordinates via a generative diffusion process.
    It leverages a diffusion model trained to denoise 'noised' atomic coordinates back to their
    true state. The diffusion process captures both local and global structural information
    through a series of noise scales.

    Attributes:
        channels (int): The number of channels in the input feature map, corresponding to atomic features.
        num_diffusion_steps (int): The number of diffusion steps or noise levels to use.
    """

    def __init__(
        self,
        channels: int,
        num_diffusion_steps: int = 1000,
        training: bool = False,
        depth: int = 30,
    ):
        """
        Initializes the DiffusionModule with the specified number of channels and diffusion steps.

        Args:
            channels (int): Number of feature channels for the input.
            num_diffusion_steps (int): Number of diffusion steps (time steps in the diffusion process).
        """
        super(GeneticDiffusionModuleBlock, self).__init__()
        self.channels = channels
        self.num_diffusion_steps = num_diffusion_steps
        self.training = training
        self.depth = depth
        self.noise_scale = nn.Parameter(
            torch.linspace(1.0, 0.01, num_diffusion_steps)
        )
        self.prediction_network = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: Tensor = None, ground_truth: Tensor = None):
        """
        Forward pass of the DiffusionModule. Applies a sequence of noise and denoising operations to
        the input coordinates to simulate the diffusion process.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_atoms, channels)
                            representing the atomic features including coordinates.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_atoms, channels) with
                        denoised atom coordinates.
        """
        batch_size, num_nodes, num_nodes_two, num_features = x.shape
        noisy_x = x.clone()

        for step in range(self.num_diffusion_steps):
            # Generate noise scaled by the noise level for the current step
            noise_level = self.noise_scale[step]
            noise = torch.randn_like(x) * noise_level

            # Add noise to the input
            noisy_x = x + noise

            # Predict and denoise the noisy input
            noisy_x = self.prediction_network(noisy_x)

        if self.training and ground_truth is not None:
            loss = F.mse_loss(noisy_x, ground_truth)
            return noisy_x, loss

        return noisy_x


class GeneticDiffusion(nn.Module):
    """
    GeneticDiffusion module for performing genetic diffusion.

    Args:
        channels (int): Number of input channels.
        num_diffusion_steps (int): Number of diffusion steps to perform.
        training (bool): Whether the module is in training mode or not.
        depth (int): Number of diffusion module blocks to stack.

    Attributes:
        channels (int): Number of input channels.
        num_diffusion_steps (int): Number of diffusion steps to perform.
        training (bool): Whether the module is in training mode or not.
        depth (int): Number of diffusion module blocks to stack.
        layers (nn.ModuleList): List of GeneticDiffusionModuleBlock instances.

    """

    def __init__(
        self,
        channels: int,
        num_diffusion_steps: int = 1000,
        training: bool = False,
        depth: int = 30,
    ):
        super(GeneticDiffusion, self).__init__()
        self.channels = channels
        self.num_diffusion_steps = num_diffusion_steps
        self.training = training
        self.depth = depth

        # Layers
        self.layers = nn.ModuleList(
            [
                GeneticDiffusionModuleBlock(
                    channels, num_diffusion_steps, training, depth
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor = None, ground_truth: Tensor = None):
        """
        Forward pass of the GeneticDiffusion module.

        Args:
            x (Tensor): Input tensor.
            ground_truth (Tensor): Ground truth tensor.

        Returns:
            Tuple[Tensor, Tensor]: Output tensor and loss tensor.

        """
        for layer in self.layers:
            x, loss = layer(x, ground_truth)
        return x, loss


# # Example usage
# if __name__ == "__main__":
#     model = GeneticDiffusionModuleBlock(
#         channels=3, training=True
#     )  # Assuming 3D coordinates
#     input_coords = torch.randn(
#         10, 100, 100, 3
#     )  # Example with batch size 10 and 100 atoms
#     ground_truth = torch.randn(
#         10, 100, 100, 3
#     )  # Example with batch size 10 and 100 atoms
#     output_coords, loss = model(input_coords, ground_truth)
#     print(output_coords)  # Should be (10, 100, 3)
#     print(loss)  # Should be a scalar (MSE loss value
